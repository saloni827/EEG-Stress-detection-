import zipfile
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === Step 1: Extract Dataset ===
zip_path = '/content/drive/MyDrive/archive.zip'
extract_path = '/content/EEG_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# === Step 2: Load CSV Files into Dictionary by Task ===
data_paths = {
    "Stroop": [],
    "Relax": [],
    "Arithmetic": [],
    "Mirror_image": []
}

for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith(".csv"):
            full_path = os.path.join(root, file)
            for task in data_paths.keys():
                if task in full_path:
                    data_paths[task].append(full_path)

# === Step 3: Task Labels (for Binary Classification) ===
task_labels = {
    'Relax': 0,
    'Mirror_image': 1,
    'Arithmetic': 1,
    'Stroop': 1
}

# === Step 4: Preprocessing Functions ===
def bandpass_filter(data, lowcut=0.5, highcut=50, fs=256):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    if len(data) > 3 * max(len(a), len(b)):
        return filtfilt(b, a, data)
    else:
        return data

def preprocess_data(file_path, fs=256, window_size=2):
    df = pd.read_csv(file_path)
    eeg_channels = [col for col in df.columns if col not in ['Time', 'timestamp', 'label']]
    
    # Filter
    for ch in eeg_channels:
        df[ch] = bandpass_filter(df[ch].values, fs=fs)

    # Normalize
    scaler = MinMaxScaler()
    df[eeg_channels] = scaler.fit_transform(df[eeg_channels])

    # Segment
    segment_size = fs * window_size
    segments = []
    for start in range(0, len(df) - segment_size, segment_size // 2):
        segment = df.iloc[start:start + segment_size][eeg_channels].values
        segments.append(segment)

    return np.array(segments)

# === Step 5: Load, Preprocess and Label Data ===
X, y = [], []

for task, file_list in data_paths.items():
    label = task_labels[task]
    for file in file_list:
        try:
            segments = preprocess_data(file)
            X.extend(segments)
            y.extend([label] * len(segments))
            print(f"Processed {file} -> {segments.shape}")
        except Exception as e:
            print(f"Error in {file}: {e}")

X = np.array(X)
y = np.array(y)
print("Final Data Shapes ->", X.shape, y.shape)

# === Step 6: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Step 7: Create Enhanced CNN Model ===
def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# === Step 8: Train the Model ===
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# === Step 9: Evaluate and Visualize ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {test_acc:.4f}")

# Plot Accuracy and Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
