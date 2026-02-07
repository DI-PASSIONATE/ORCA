import copy
from typing import Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import tqdm
import optuna


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
    patience=20,
    criterion=nn.MSELoss(),
    optimizer=AdamW,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    progress_callback=None,
    stage_name="Training",
    trial: optuna.trial.Trial|None = None,
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
        val_loss = _val(model, epoch, criterion, device, val_loader, scheduler, trial)

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


def _val(model, epoch, criterion, device, val_loader, scheduler, trial: optuna.trial.Trial|None):
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

    if trial:
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
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

def hyperparameter_tuning(train_dataset, val_dataset, geometry) -> dict[str, Any]:
    """
    Performs hyperparameter tuning using optuna to find the best hyperparameters for the model.
    This method can be called within the run method if self.hyperparameters is None to perform tuning.
    """
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
        
        # Instantiate model
        try:
            model = geometry.get_model(hyperparameters)
        except Exception:
            import traceback
            traceback.print_exc()
            raise optuna.exceptions.TrialPruned()

        # Extract training parameters from hyperparameters
        epochs = hyperparameters.get("epochs", 50)
        batch_size = hyperparameters.get("batch_size", 32)
        learning_rate = hyperparameters.get("learning_rate", 1e-3)

        model, val_loss = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            progress_callback=None,
            stage_name="Tuning",
            trial=trial,
        )
        return val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=50)

    return study.best_params
    