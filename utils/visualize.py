import matplotlib.pyplot as plt
import random
import os
import torch
import config

def plot_loss_curve(train_losses, val_losses, path='results/loss_curve.png'):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("CNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)

def compare_models(cnn_scores, svm_scores, path='results/model_comparison.png'):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    cnn_vals = [cnn_scores[m] for m in metrics]
    svm_vals = [svm_scores[m] for m in metrics]

    x = range(len(metrics))
    plt.figure()
    plt.bar([i - 0.2 for i in x], cnn_vals, width=0.4, label='CNN')
    plt.bar([i + 0.2 for i in x], svm_vals, width=0.4, label='SVM')
    plt.xticks(ticks=x, labels=metrics)
    plt.title("CNN vs SVM Performance")
    plt.legend()
    plt.savefig(path)

def plot_mel_spectrogram(cache_dir=config.CACHE_DIR, save_path=None):
    pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
    if not pt_files:
        print("No cached spectrograms found in the directory.")
        return

    # Pick a random sample
    selected_file = random.choice(pt_files)
    spec_tensor = torch.load(os.path.join(cache_dir, selected_file))  # shape: (1, 128, 128)
    mel = spec_tensor.squeeze().numpy()  # shape: (128, 128)

    # Plot the mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram - {selected_file}")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved spectrogram to {save_path}")
    else:
        plt.show()
