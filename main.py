import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

# --------------------------------------------------
# 데이터 로딩 및 전처리
# --------------------------------------------------

csv_path = 'dynamic_api_call_sequence_per_malware_100_0_306.csv'
df = pd.read_csv(csv_path)

X = df[[f't_{i}' for i in range(100)]].values
y = df['malware'].values

X_train = torch.tensor(X, dtype=torch.long)
y_train = torch.tensor(y, dtype=torch.float32)

VOCAB_SIZE = int(X_train.max()) + 1

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------------------------------
# device 설정 (GPU 또는 CPU 자동 선택)
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# --------------------------------------------------
# 모델 정의 (가벼운 버전)
# --------------------------------------------------

EMBED_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 2
FFN_HIDDEN = 64
EPOCHS = 10
LR = 0.001
MAX_LEN = 100

class MalwareDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_hidden, max_len):
        super(MalwareDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        x = (x * attn_weights).sum(dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
'''
model = MalwareDetector(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_HIDDEN, MAX_LEN).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} 시작")
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
'''
# --------------------------------------------------
# 모델 평가 (조기 탐지 실험 지원)
# --------------------------------------------------

def k_fold_cross_validation(X, y, k=5, short_len=None):
    """
    k-fold 교차검증을 수행하는 함수
    short_len을 설정하면, 테스트 데이터에 한해 시퀀스 앞 short_len개만 사용 (조기 탐지 실험용)
    훈련 데이터는 항상 전체 시퀀스를 사용함
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    all_reports = []

    for train_index, test_index in kf.split(X):
        fold += 1
        print(f"\nFold {fold} 시작")

        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # ✅ 조기 탐지: 테스트 데이터만 앞 short_len개 + 패딩
        if short_len is not None and short_len < MAX_LEN:
            print(f"[조기 탐지] 테스트 입력: 앞 {short_len}개만 사용 + 0 패딩")
            X_test_fold = X_test_fold[:, :short_len]
            pad_len = MAX_LEN - short_len
            X_test_fold = torch.cat(
                [X_test_fold, torch.zeros((X_test_fold.shape[0], pad_len), dtype=torch.long)],
                dim=1
            )

        dataset_train = TensorDataset(X_train_fold, y_train_fold)
        dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

        model = MalwareDetector(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_HIDDEN, MAX_LEN).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        model.train()
        for epoch in range(EPOCHS):
            for batch_X, batch_y in dataloader_train:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            X_test_fold = X_test_fold.to(device)
            predictions = model(X_test_fold).squeeze().cpu().numpy()
        predicted_labels = (predictions >= 0.5).astype(int)
        report = classification_report(
            y_test_fold.numpy(), predicted_labels, target_names=["정상", "악성"], output_dict=True, zero_division=0
        )
        all_reports.append(report)

        print(f"Fold {fold} 결과:")
        for label, metrics in report.items():
            if label in ["정상", "악성"]:
                print(f"  {label}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
            elif label == "accuracy":
                print(f"  {label}: {metrics:.4f}")
            else:
                print(f"  {label}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")

    avg_report = {
        "정상": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
        "악성": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
        "accuracy": 0,
        "macro avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
        "weighted avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
    }
    for report in all_reports:
        for key in avg_report.keys():
            if key in report:
                if key == "accuracy":
                    avg_report[key] += report[key]
                else:
                    for metric in avg_report[key].keys():
                        avg_report[key][metric] += report[key][metric]
    for key in avg_report.keys():
        if key == "accuracy":
            avg_report[key] /= k
        else:
            for metric in avg_report[key].keys():
                avg_report[key][metric] /= k

    print("\nAverage Report:")
    for key, metrics in avg_report.items():
        if key == "accuracy":
            print(f"{key}: {metrics:.4f}")
        else:
            print(f"{key}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

# (1) 전체 시퀀스 사용 (기존 방식)
# k_fold_cross_validation(X_train, y_train)

# (2) 조기 탐지 실험 1 (앞 20개만 사용)
k_fold_cross_validation(X_train, y_train, short_len=20)

# (3) 조기 탐지 실험 2 (앞 50개만 사용)
k_fold_cross_validation(X_train, y_train, short_len=50)