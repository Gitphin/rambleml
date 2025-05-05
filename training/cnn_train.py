import torch

def train_cnn(data_loader, model, criterion, optimizer, epochs=10, device='cuda'):
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Logging the loss at the end of each epoch
        train_losses.append(running_loss / len(data_loader))

        # Validation Loss (this could be done with a separate validation dataset)
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_losses.append(val_loss / len(data_loader))
            model.train()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

