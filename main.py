import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.loader import EmotionDataset
from models.cnn import AudioCNN
from utils.metrics import evaluate_metrics
from training.svm_train import train_svm
from training.cnn_train import train_cnn
from utils.visualize import plot_loss_curve, compare_models, plot_mel_spectrogram
import config


def main():
    # Settings
    data_dir = config.DATA_DIR
    emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Suprised"]
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCHS
    device = config.DEVICE
    plot_mel_spectrogram(cache_dir="data", save_path="results/sample_mel.png")

    # Load dataset
    dataset = EmotionDataset(data_dir=data_dir, emotions=emotions, cache_dir=config.CACHE_DIR)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize CNN model
    cnn_model = AudioCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=config.LEARNING_RATE)

    # Train CNN
    train_losses, val_losses = train_cnn(data_loader, cnn_model, criterion, optimizer, epochs=epochs, device=device)
    
    # Plot CNN loss curve
    plot_loss_curve(train_losses, val_losses, path='results/loss_curve.png')

    # Evaluate CNN
    cnn_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn_model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    cnn_scores = evaluate_metrics(y_true, y_pred)

    # Train SVM
    svm_scores = train_svm()

    # Compare CNN and SVM performance
    compare_models(cnn_scores, svm_scores, path='results/model_comparison.png')

    print("\nCNN Evaluation Metrics:")
    print(cnn_scores)
    print("\nSVM Evaluation Metrics:")
    print(svm_scores)

if __name__ == "__main__":
    main()

