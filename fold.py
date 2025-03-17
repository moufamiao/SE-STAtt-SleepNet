import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from tqdm import tqdm
import pre_data.loadedf
import pre_data.loadedf_78
from imblearn.metrics import geometric_mean_score


# 自定义数据预处理模块 ------------------------------------------------------
class EEGPreprocessor:
    def __init__(self, train_mean=None, train_std=None):
        self.train_mean = train_mean
        self.train_std = train_std

    def wavelet_denoise(self, signal):
        """小波去噪处理"""
        coeffs = pywt.wavedec(signal, 'db4', level=5)
        coeffs[1:] = [pywt.threshold(c, value=0.1 * np.max(c), mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, 'db4')

    def normalize(self, X, eps=1e-8):
        """基于训练集的标准化"""
        if self.train_mean is None:
            self.train_mean = np.mean(X, axis=(0, 1))
            self.train_std = np.std(X, axis=(0, 1)) + eps
        return (X - self.train_mean) / self.train_std


# 增强型数据集类 -----------------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y, preprocessor=None, augment=False):
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx]

        # 小波去噪
        denoised = np.stack([self.preprocessor.wavelet_denoise(ch) for ch in signal.T], axis=1)

        # 数据增强
        if self.augment:
            shift = np.random.randint(-100, 100)
            denoised = np.roll(denoised, shift, axis=0)
            noise = np.random.normal(0, 0.05, denoised.shape)
            denoised = denoised + noise

        return torch.FloatTensor(denoised), torch.LongTensor([self.y[idx]]).squeeze()


# 模型定义保持不变...
class EEGHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 64, kernel_size=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            dropout=0.6,
            batch_first=True
        )

        # 注意力机制（输入维度改为 128）
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 分类器（输入维度改为 128）
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(128, 5)  # 输出 5 类
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # CNN处理
        x = x.permute(0, 2, 1)  # [batch, channels, time]
        x = self.cnn(x)  # [batch, 64, seq]
        x = x.permute(0, 2, 1)  # [batch, seq, features]

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [batch, seq, 128]

        # 注意力机制
        attn_weights = self.attention(lstm_out)  # [batch, seq, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, 128]

        # 分类
        return self.classifier(context)


class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def train(self, train_loader, val_loader, epochs, fold_idx):
        optimizer = optim.AdamW([
            {'params': self.model.cnn.parameters(), 'lr': 1e-4},
            {'params': self.model.lstm.parameters(), 'lr': 1e-3},
            {'params': self.model.classifier.parameters(), 'lr': 5e-3}
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_model = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            progress = tqdm(train_loader, desc=f'Fold {fold_idx} Epoch {epoch + 1}/{epochs}')
            for inputs, labels in progress:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                progress.set_postfix({'loss': loss.item()})

            val_loss, val_acc = self.evaluate(val_loader)
            scheduler.step(val_loss)

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = self.model.state_dict()

            print(f'Fold {fold_idx} Epoch {epoch + 1}: '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # 加载最佳模型
        self.model.load_state_dict(best_model)
        return best_acc

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        return avg_loss, accuracy

    def predict(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        return all_preds, all_labels


# 主程序 ----------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = pre_data.loadedf.loaddata()
    X = np.transpose(X, (0, 2, 1))

    skf = StratifiedKFold(n_splits=20, shuffle=True)
    all_preds = []
    all_labels = []
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Processing Fold {fold_idx + 1}/20 ===")

        # 新建预处理器
        preprocessor = EEGPreprocessor()

        # 仅用训练集计算参数
        X_train_fold = X[train_idx]
        X_train_norm = preprocessor.normalize(X_train_fold)

        # 验证集使用训练集参数
        X_val_norm = (X[val_idx] - preprocessor.train_mean) / preprocessor.train_std

        # 创建数据集
        train_dataset = EEGDataset(X_train_norm, y[train_idx], preprocessor, augment=True)
        val_dataset = EEGDataset(X_val_norm, y[val_idx], preprocessor, augment=False)

        # 创建数据加载器
        batch_size = 512
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        # 初始化模型和训练器
        model = EEGHybridModel()
        trainer = Trainer(model, device)

        # 训练并获取最佳准确率
        best_acc = trainer.train(train_loader, val_loader, epochs=100, fold_idx=fold_idx + 1)
        fold_accuracies.append(best_acc)

        # 收集预测结果
        fold_preds, fold_labels = trainer.predict(val_loader)
        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)

    # 整体评估
    print("\n=== Final Evaluation ===")
    print(f"20-Fold Cross Validation Accuracies: {fold_accuracies}")
    print(f"Mean Validation Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"f1_score: {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(all_labels, all_preds):.4f}")
    print(f"Macro-averaged G-mean: {geometric_mean_score(all_labels, all_preds, average='macro'):.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 获取类别名称（假设类别从0开始）
    classes = np.arange(cm.shape[0])

    # 打印表头
    print(f"{'':<7} |", end="")
    for cls in classes:
        print(f"{'Predicted ' + str(cls):<9}", end="")
    print(f" | {'Total':<6} | {'Accuracy':<8}")
    print("-" * (10 + 11 * len(classes)))

    # 打印每一行
    for i, cls in enumerate(classes):
        true_count = np.sum(cm[i, :])
        accuracy = cm[i, i] / true_count if true_count > 0 else 0
        print(f"True {cls:<3} |", end="")
        for j in range(len(classes)):
            print(f"{cm[i, j]:^9}", end="")
        print(f" | {true_count:^6} | {accuracy:^8.2%}")

    # 打印总计数
    print("-" * (10 + 11 * len(classes)))
    print(f"{'Total':<7} |", end="")
    for j in range(len(classes)):
        col_count = np.sum(cm[:, j])
        print(f"{col_count:^9}", end="")
    print()

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

