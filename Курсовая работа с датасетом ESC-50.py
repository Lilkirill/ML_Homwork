import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. DATALOADER CLASS AND DATASET PREPARATION
class ESCDataset(Dataset):
    def __init__(self, dataset_dir, labels_path, transform=None):
        self.dataset_dir = dataset_dir
        self.labels = pd.read_csv(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_filename = self.labels.iloc[idx, 0]
        audio_label = self.labels.iloc[idx, 2]  # Второй столбец содержит метку класса
        audio_path = os.path.join(self.dataset_dir, audio_filename)
        audio_data, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
        audio_data = audio_data[:16000] if len(audio_data) > 16000 else np.pad(audio_data, (0, max(0, 16000 - len(audio_data))), "constant")
        if self.transform:
            audio_data = self.transform(audio_data)
        return torch.tensor(audio_data, dtype=torch.float32), torch.tensor(audio_label, dtype=torch.long), audio_filename


#Загрузка набора данных
ESC50_audio_dir = r"C:\Users\kirill.nikitin\Downloads\ESC-50-master\ESC-50-master\audio"  # Путь к папке с аудиофайлами ESC-50
ESC50_labels_path = r"C:\Users\kirill.nikitin\Downloads\ESC-50-master\ESC-50-master\meta\esc50.csv"  # Путь к мета-файлу

ESC50_dataset = ESCDataset(ESC50_audio_dir, ESC50_labels_path)

# Разделение данных
train_indices, validation_indices = train_test_split(range(len(ESC50_dataset)), test_size=0.2, random_state=42)
train_data_subset = torch.utils.data.Subset(ESC50_dataset, train_indices)
validation_data_subset = torch.utils.data.Subset(ESC50_dataset, validation_indices)

train_data_loader = DataLoader(train_data_subset, batch_size=16, shuffle=True)
validation_data_loader = DataLoader(validation_data_subset, batch_size=16, shuffle=False)

# Отображение нескольких примеров из dataloader
for batch_index, (audio_batch, label_batch, filenames_batch) in enumerate(train_data_loader):
    print(f"Пакет {batch_index + 1}")
    print(f"Форма аудиоданных: {audio_batch.shape}")
    print(f"Метки: {label_batch}")
    print(f"Имена файлов: {filenames_batch}")
    if batch_index == 2:  # Отобразить только первые 3 пакета
        break

# 2. STATISTICS ANALYSIS
def dataset_statistics(dataframe):
    print("Статистика набора данных:\n")
    print("Всего записей:", len(dataframe))
    print("Поля:", dataframe.columns.tolist())
    print("Типы данных:")
    print(dataframe.dtypes)
    print("Распределение классов:")
    print(dataframe['category'].value_counts(normalize=True) * 100)  # Столбец с метками классов
    print("Пропущенные значения:")
    print(dataframe.isnull().sum())
    record_size = dataframe.memory_usage(index=True).sum() / len(dataframe) * 8  # Размер одной записи в битах
    print(f"Объем одной записи (в битах): {record_size:.2f}")


labels_dataframe = pd.read_csv(ESC50_labels_path)
dataset_statistics(labels_dataframe)

# Создание словаря меток
label_to_category_map = labels_dataframe[['target', 'category']].drop_duplicates().set_index('target')['category'].to_dict()


# 3. MACHINE LEARNING MODEL - AUDIO CLASSIFICATION
class SimpleAudioClassifier(nn.Module):
    def __init__(self):
        super(SimpleAudioClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(16000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 50)  # 50 классов в ESC-50
        )

    def forward(self, x):
        return self.fc(x)

def train_model(classifier_model, training_loader, validation_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
    losses = []

    for epoch in range(epochs):
        classifier_model.train()
        total_loss = 0
        for audio_batch, label_batch, _ in training_loader:
            predictions = classifier_model(audio_batch)
            loss = criterion(predictions, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(training_loader)
        losses.append(average_loss)
        print(f"Эпоха {epoch + 1}, Потеря: {average_loss:.4f}")

    # Построение графика потерь
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), losses, marker='o', label='Потери на обучении')
    plt.title("График потерь от эпох")
    plt.xlabel("Эпоха")
    plt.ylabel("Потеря")
    plt.legend()
    plt.grid()
    plt.show()

classifier_model_instance = SimpleAudioClassifier()
train_model(classifier_model_instance, train_data_loader, validation_data_loader)


# INFERENCE FUNCTION
def inference(classifier_model, audio_sample, sample_filename, actual_label):
    classifier_model.eval()
    with torch.no_grad():
        prediction_scores = classifier_model(audio_sample.unsqueeze(0))
        predicted_label = torch.argmax(prediction_scores, dim=1).item()
    predicted_category = label_to_category_map.get(predicted_label, "Неизвестно")
    actual_category = label_to_category_map.get(actual_label, "Неизвестно")

    print(f"Название аудиоклипа: {sample_filename}")
    print(f"Предсказанная категория: {predicted_category}")

    return predicted_category

example_audio, example_label, example_filename = ESC50_dataset[0]
predicted_category_output = inference(classifier_model_instance, example_audio, example_filename, example_label)
example_audio, example_label, example_filename = ESC50_dataset[1]
predicted_category_output = inference(classifier_model_instance, example_audio, example_filename, example_label)

# 5. CLUSTERING & DIMENSION REDUCTION
def clustering_and_dim_reduction(audio_dir, n_clusters=5):
    audio_features = []
    audio_filenames = []
    for audio_filename in os.listdir(audio_dir):
        if audio_filename.endswith(".wav"):
            audio_data, sampling_rate = librosa.load(os.path.join(audio_dir, audio_filename), sr=16000, mono=True)
            mfcc_features = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
            audio_features.append(np.mean(mfcc_features, axis=1))
            audio_filenames.append(audio_filename)

    audio_features = np.array(audio_features)
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(audio_features)

    # Уменьшение размерности
    pca_model = PCA(n_components=2)
    reduced_features = pca_model.fit_transform(scaled_features)

    # Кластеризация
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans_model.fit_predict(reduced_features)

    # Создание словаря для отображения кластеров в легенду
    cluster_labels = [f"Кластер {i}" for i in range(n_clusters)]
    color_palette = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    cluster_color_mapping = {i: color for i, color in enumerate(color_palette)}

    # Визуализация
    plt.figure(figsize=(10, 7))
    for cluster_index in range(n_clusters):
        cluster_points = reduced_features[cluster_assignments == cluster_index]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_color_mapping[cluster_index], label=f"Кластер {cluster_index}")

    plt.title("Кластеризация аудиоклипов")
    plt.xlabel("Главный компонент 1")
    plt.ylabel("Главный компонент 2")
    plt.legend()
    plt.show()

clustering_and_dim_reduction(ESC50_audio_dir)



# 6. OUTLIER DETECTION
def detect_audio_outliers(audio_dir):
    audio_features = []
    for audio_filename in os.listdir(audio_dir):
        if audio_filename.endswith(".wav"):
            audio_data, sampling_rate = librosa.load(os.path.join(audio_dir, audio_filename), sr=16000, mono=True)
            mfcc_features = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
            audio_features.append(np.mean(mfcc_features, axis=1))

    audio_features = np.array(audio_features)
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(audio_features)

    # Использование IsolationForest для поиска выбросов
    isolation_forest_model = IsolationForest(contamination=0.05, random_state=42)
    isolation_outlier_predictions = isolation_forest_model.fit_predict(scaled_features)

    # Использование LocalOutlierFactor для подтверждения выбросов
    local_outlier_factor_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_outlier_predictions = local_outlier_factor_model.fit_predict(scaled_features)

    isolation_outliers = np.where(isolation_outlier_predictions == -1)[0]
    confirmed_outliers = np.where((isolation_outlier_predictions == -1) & (lof_outlier_predictions == -1))[0]

    print(f"Обнаружено выбросов (Isolation Forest): {len(isolation_outliers)}")
    print(f"Подтверждено выбросов (LOF): {len(confirmed_outliers)}")

    for outlier_index in confirmed_outliers:
        print(f"Выброс: {os.listdir(audio_dir)[outlier_index]}")

detect_audio_outliers(ESC50_audio_dir)
