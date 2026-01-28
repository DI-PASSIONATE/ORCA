import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import tqdm

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

    return 2*lcsh + mse

def train_model(
    dataset,
    model: nn.Module,
    epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    patience=20,
    criterion = nn.MSELoss(),
    optimizer = AdamW,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    progress_callback=None,
):
    model.to(device)

    train_split = dataset.get_train_split()
    val_split = dataset.get_val_split()

    train_split.load_samples()
    val_split.load_samples()

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)

    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 2

    if progress_callback:
        progress_callback("Training", 0, epochs, "Starting training")

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
                "Training",
                epoch + 1,
                epochs,
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f}",
            )

        print(
            f"Epoch {epoch+1:4d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

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
    dataset,
    model: nn.Module,
    batch_size=32,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_split = dataset.get_test_split()
    test_split.load_samples()

    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False)

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