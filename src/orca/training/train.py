import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

def train_model(dataset, model: nn.Module, epochs=10, batch_size=32, learning_rate=1e-3, progress_callback=None) -> nn.Module:
    """
    Trains the provided model using the given dataset.

    Args:
        dataset: The training dataset.
        model (nn.Module): The neural network model to be trained.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Learning rate for the optimizer.
        progress_callback (callable, optional): A callback function to report progress. It should accept four arguments: step, current, total, message.
    Returns:
        nn.Module: The trained model.
    """

    # Load dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()

    if progress_callback:
        progress_callback("Model Training", 0, epochs, "Starting model training...")

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss_value = loss(pred, targets)
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.item()
        avg_loss = total_loss / len(dataloader)
        if progress_callback:
            progress_callback("Model Training", epoch + 1, epochs, f"Avg Loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model