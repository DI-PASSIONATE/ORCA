import copy
from typing import Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import tqdm
import optuna
from sklearn.model_selection import KFold


def complex_mse(pred, target):
    N = pred.shape[1] // 2
    pred = pred.view(-1, N, 2)
    target = target.view(-1, N, 2)
    return torch.mean((pred - target) ** 2)


def mse_plus_log_cosh_loss(pred, target):
    N = pred.shape[1] // 2
    diff = pred - target
    lcsh = torch.mean(torch.log(torch.cosh(diff)))
    pred = pred.view(-1, N, 2)
    target = target.view(-1, N, 2)
    mse = torch.mean((pred - target) ** 2)

    return 2 * lcsh + mse


def train_model(
    train_dataset,
    val_dataset,
    model: nn.Module,
    epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    patience=10,
    criterion=nn.L1Loss(),
    optimizer=AdamW,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    progress_callback=None,
    stage_name="Training",
):
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optimizer(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 2

    if progress_callback:
        progress_callback(stage_name, 0, epochs, "Starting training")

    for epoch in range(epochs):
        # ---- TRAIN ----
        train_loss = _train(model, criterion, optimizer, device, train_loader)

        # ---- VALIDATION ----
        val_loss = _val(model, criterion, device, val_loader, scheduler)

        # ---- EARLY STOPPING ----
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if progress_callback:
            progress_callback(
                stage_name,
                epoch + 1,
                epochs,
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f}",
            )

        print(f"Epoch {epoch + 1:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss


def _train(model, criterion, optimizer, device, train_loader):
    model.train()
    train_loss = 0.0

    for x, y in tqdm.tqdm(train_loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    return train_loss


def _val(model, criterion, device, val_loader, scheduler):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)
        
    return val_loss


def test_model(
    test_dataset,
    model: nn.Module,
    batch_size=32,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = complex_mse

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm.tqdm(test_loader, desc="Testing", leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += criterion(pred, y).item()

    test_loss /= len(test_loader)

    return test_loss

def hyperparameter_tuning(train_val_df, result_dir, geometry, n_fold_cv: int = 5) -> dict[str, Any]:
    """
    Performs hyperparameter tuning using optuna to find the best hyperparameters for the model.
    Uses n-fold cross-validation on the provided train_val_df.
    
    Args:
        train_val_df: The dataframe to use for training and validation.
        result_dir: Directory path for dataset creation.
        geometry: The geometry instance with get_model and get_hyperparameter_search_space methods.
        n_fold_cv: Number of folds for cross-validation (default: 5).
    
    Returns:
        A dictionary of the best hyperparameters found.
    """
    # Create dataset from train_val_df to use for k-fold CV
    train_val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=train_val_df)
    
    def objective(trial):
        hyperparameters = {}
        for key, distribution in geometry.get_hyperparameter_search_space().items():
            if isinstance(distribution, list):
                hyperparameters[key] = trial.suggest_categorical(key, distribution)
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                hyperparameters[key] = trial.suggest_int(key, distribution.low, distribution.high, step=distribution.step)
            elif isinstance(distribution, optuna.distributions.FloatDistribution):
                hyperparameters[key] = trial.suggest_float(key, distribution.low, distribution.high, step=distribution.step)
            elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                 hyperparameters[key] = trial.suggest_categorical(key, distribution.choices)
        
        # Extract training parameters from hyperparameters
        epochs = hyperparameters.get("epochs", 50)
        batch_size = hyperparameters.get("batch_size", 32)
        learning_rate = hyperparameters.get("learning_rate", 1e-3)
        
        # Perform k-fold cross-validation on train_val set only
        kfold = KFold(n_splits=n_fold_cv, shuffle=True, random_state=42)
        fold_losses = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(train_val_dataset)):
            # Create fold datasets (convert numpy arrays to lists)
            fold_train_dataset = Subset(train_val_dataset, list(train_indices))
            fold_val_dataset = Subset(train_val_dataset, list(val_indices))
            
            # Instantiate model for this fold
            try:
                model = geometry.get_model(hyperparameters)
            except Exception:
                import traceback
                traceback.print_exc()
                raise optuna.exceptions.TrialPruned()
            
            # Train model on this fold
            try:
                model, fold_val_loss = train_model(
                    train_dataset=fold_train_dataset,
                    val_dataset=fold_val_dataset,
                    model=model,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    progress_callback=None,
                    stage_name=f"Tuning (Fold {fold_idx + 1}/{n_fold_cv})",
                )
                fold_losses.append(fold_val_loss)

                # Report once per fold using a unique step index for pruning.
                trial.report(fold_val_loss, fold_idx)
                print(f"Fold {fold_idx + 1}/{n_fold_cv} | Val Loss: {fold_val_loss:.4f}")
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise optuna.exceptions.TrialPruned()
        
        # Return average validation loss across folds
        avg_loss = sum(fold_losses) / len(fold_losses)
        return avg_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=200)

    return study.best_params
    