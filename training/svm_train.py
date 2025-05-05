import os
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.metrics import evaluate_metrics
import config

def load_cached_spectrograms(cache_dir=config.CACHE_DIR):
    X, y = [], []
    for file in os.listdir(cache_dir):
        if file.endswith(".pt"):
            emotion = file.split("_")[0]
            label = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Suprised"].index(emotion)
            spec = torch.load(os.path.join(cache_dir, file)).numpy().flatten()
            X.append(spec)
            y.append(label)
    return np.array(X), np.array(y)

def train_svm():
    X, y = load_cached_spectrograms()
    X = StandardScaler().fit_transform(X)
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=2/3, random_state=42)

    model = SVC(kernel="rbf", C=1, gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_metrics(y_test, y_pred)

