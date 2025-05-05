# config.py

import os

DATA_DIR = 'data/Emotions'
CACHE_DIR = 'data/cached_data'
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 5  # seconds (originally 3)
NUM_CLASSES = 7
INPUT_SIZE = (1, 128, 128)

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

SEED = 42
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

