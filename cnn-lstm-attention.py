import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, cohen_kappa_score,confusion_matrix, f1_score
import matplotlib.pyplot as plt
import pywt
from tqdm import tqdm
import pre_data.loadedf_shhs
import pre_data.loadedf_78
import pre_data.loadedf
from imblearn.metrics import geometric_mean_score

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
        signal = self.X[idx]  # 已预处理和标准化的数据

        # 数据增强
        if self.augment:
            # 随机时间偏移
            shift = np.random.randint(-100, 100)
            signal = np.roll(signal, shift, axis=0)
            # 添加高斯噪声
            noise = np.random.normal(0, 0.05, signal.shape)
            signal = signal + noise

        # 返回信号和标签
        return torch.FloatTensor(signal), torch.LongTensor([self.y[idx]]).squeeze()  # 确保标签为标量

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
            nn.MaxPool1d(3),
            nn.Dropout(0.3)
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
        self.lstm_norm = nn.LayerNorm(128)

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
        lstm_out = self.lstm_norm(lstm_out)
        # 注意力机制
        attn_weights = self.attention(lstm_out)  # [batch, seq, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, 128]

        # 分类
        return self.classifier(context)

class Trainer:
    def __init__(self, model, device, preprocessor):
        self.model = model.to(device)
        self.device = device
        self.preprocessor = preprocessor

    def prepare_data(self, X, y):
        """修正后的数据预处理流程"""
        # 步骤1：先划分原始数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.05,
            stratify=y,
            random_state=42  # 可重复性种子
        )

        # 步骤2：对原始数据进行小波去噪
        temp_preprocessor = EEGPreprocessor()
        # 处理训练集
        X_train_denoised = np.array([
            np.stack([temp_preprocessor.wavelet_denoise(ch) for ch in sample.T], axis=1)
            for sample in X_train
        ])
        # 处理验证集
        X_val_denoised = np.array([
            np.stack([temp_preprocessor.wavelet_denoise(ch) for ch in sample.T], axis=1)
            for sample in X_val
        ])

        # 步骤3：用去噪后的训练数据计算标准化参数
        self.preprocessor = EEGPreprocessor()
        X_train_norm = self.preprocessor.normalize(X_train_denoised)  # 计算并应用标准化

        # 步骤4：验证集使用训练参数标准化
        X_val_norm = (X_val_denoised - self.preprocessor.train_mean) / self.preprocessor.train_std

        # 步骤5：创建数据集
        train_dataset = EEGDataset(X_train_norm, y_train, self.preprocessor, augment=True)
        val_dataset = EEGDataset(X_val_norm, y_val, self.preprocessor, augment=False)

        return train_dataset, val_dataset

    def train(self, train_loader, val_loader, epochs):
        # 优化配置
        optimizer = optim.AdamW([
            {'params': self.model.cnn.parameters(), 'lr': 1e-4},
            {'params': self.model.lstm.parameters(), 'lr': 1e-3},
            {'params': self.model.classifier.parameters(), 'lr': 5e-3}
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        # 训练记录
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0.0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for inputs, labels in progress:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                progress.set_postfix({'loss': loss.item()})

            # 验证阶段
            val_loss, val_acc, val_metrics = self.evaluate(val_loader)

            # 学习率调整
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

            # 记录历史
            history['train_loss'].append(train_loss / len(train_loader.dataset))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Epoch {epoch + 1}: '
                  f'Train Loss: {history["train_loss"][-1]:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f"Acc: {val_acc:.4f} | "
                  f"mF1: {val_metrics['mf1']:.4f} | "
                  f"Kappa: {val_metrics['k']:.4f} | "
                  f"G-mean: {val_metrics['mgm']:.4f} | "
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        return history

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
                loss = criterion(outputs, labels.squeeze())

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels.squeeze()).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        # 计算损失和准确率
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        mf1 = f1_score(all_labels, all_preds, average='macro')
        k = cohen_kappa_score(all_labels, all_preds)
        mgm = geometric_mean_score(all_labels, all_preds, average='macro')

        # 保持原有返回结构 + 新增指标
        return avg_loss, accuracy, {
            'mf1': mf1,
            'k': k,
            'mgm': mgm
        }


# 主程序 ----------------------------------------------------------------
if __name__ == "__main__":
    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # 加载数据
    X, y = pre_data.loadedf.loaddata()
    X = np.transpose(X, (0, 2, 1))

    # 数据预处理
    preprocessor = EEGPreprocessor()
    trainer = Trainer(EEGHybridModel(), device, preprocessor)
    train_dataset, val_dataset = trainer.prepare_data(X, y)

    # 创建DataLoader
    batch_size = 512
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )

    # 训练模型
    history = trainer.train(train_loader, val_loader, epochs=100)

    # 可视化结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()
