import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram
import config

class EmotionDataset(Dataset):
    def __init__(self, data_dir, emotions, transform=None, cache_dir=config.CACHE_DIR):
        self.data = []
        self.labels = []
        self.emotions = emotions
        self.transform = transform
        self.cache_dir = cache_dir
        self.sample_rate = config.SAMPLE_RATE
        self.mel_transform = MelSpectrogram(sample_rate=self.sample_rate, n_fft=512, hop_length=512, n_mels=config.N_MELS)
        
        os.makedirs(self.cache_dir, exist_ok=True)

        for idx, emotion in enumerate(emotions):
            folder = os.path.join(data_dir, emotion)
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    wav_path = os.path.join(folder, file)
                    cache_path = os.path.join(self.cache_dir, f"{emotion}_{file}.pt")

                    if os.path.exists(cache_path):
                        spec = torch.load(cache_path)
                    else:
                        waveform, sr = torchaudio.load(wav_path)
                        if sr != self.sample_rate:
                            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                        spec = self.mel_transform(waveform)
                        spec = torch.nn.functional.interpolate(spec.unsqueeze(0), size=(config.N_MELS, 128), mode="bilinear", align_corners=False).squeeze(0)
                        torch.save(spec, cache_path)

                    self.data.append(spec.unsqueeze(0))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        x = self.data[idx]
        assert x.shape == (1, 128, 128), f"Sample {idx} has shape {x.shape}, expected (1, 128, 128)"
        return x, self.labels[idx]

