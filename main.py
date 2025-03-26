import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

# 1. 예시 API 호출 로그 데이터
api_sequences = [
    # 악성코드 샘플 (랜섬웨어, 인젝션, 파일 조작)
    ["NtCreateFile", "NtWriteFile", "NtCloseFile", "NtDeleteFile"],  # 파일 암호화, 삭제
    ["NtOpenProcess", "NtAllocateVirtualMemory", "NtWriteVirtualMemory", "NtCreateThreadEx"],  # 프로세스 인젝션
    ["CreateRemoteThread", "VirtualAllocEx", "WriteProcessMemory", "SetThreadContext"],  # 원격 스레드 실행
    ["NtQuerySystemInformation", "NtQueryInformationProcess", "NtSetInformationThread"],  # 안티 디버깅
    
    # 정상 프로그램 샘플 (일반 파일 & 네트워크 접근)
    ["CreateFile", "ReadFile", "WriteFile", "CloseHandle"],  # 일반 파일 입출력
    ["socket", "connect", "send", "recv", "closesocket"],  # 네트워크 통신 (정상)
    ["RegOpenKeyEx", "RegQueryValueEx", "RegCloseKey"],  # 정상적인 레지스트리 접근
    ["OpenProcess", "GetModuleFileNameEx", "EnumProcesses", "CloseHandle"],  # 시스템 모니터링 도구
    ["HttpOpenRequest", "HttpSendRequest", "InternetReadFile", "InternetCloseHandle"],  # 웹 요청

    # 악성코드 샘플 (키로거, 권한 상승)
    ["SetWindowsHookEx", "GetAsyncKeyState", "SendInput"],  # 키로거 동작
    ["AdjustTokenPrivileges", "LookupPrivilegeValue", "OpenProcessToken"],  # 권한 상승
    ["NtLoadDriver", "NtUnloadDriver", "NtOpenFile"],  # 커널 드라이버 로딩
    ["CryptEncrypt", "CryptDecrypt", "CryptGenKey"],  # 랜섬웨어 암호화 API
    
    # 정상 프로그램 샘플 (서비스 실행, 윈도우 UI 조작)
    ["OpenService", "StartService", "ControlService", "CloseServiceHandle"],  # 윈도우 서비스 실행
    ["FindWindow", "SendMessage", "PostMessage"],  # UI 조작 (정상 프로그램)

    # 악성코드 샘플
    ["NtCreateFile", "NtWriteFile", "NtCloseFile"],  # 파일 암호화
    ["NtOpenProcess", "NtAllocateVirtualMemory", "NtWriteVirtualMemory"],  # 프로세스 인젝션
    ["CreateRemoteThread", "VirtualAllocEx", "WriteProcessMemory"],  # 원격 스레드 실행
    ["NtQuerySystemInformation", "NtQueryInformationProcess"],  # 안티 디버깅

    # 정상 프로그램 샘플
    ["CreateFile", "ReadFile", "WriteFile"],  # 일반 파일 입출력
    ["socket", "connect", "send", "recv"],  # 네트워크 통신 (정상)
    ["RegOpenKeyEx", "RegQueryValueEx"],  # 정상적인 레지스트리 접근
    ["OpenProcess", "GetModuleFileNameEx"],  # 시스템 모니터링 도구
    ["HttpOpenRequest", "HttpSendRequest"],  # 웹 요청

    # 악성코드 샘플
    ["NtCreateFile", "NtWriteFile", "NtCloseFile", "NtDeleteFile", "NtOpenProcess"],  # 파일 암호화 및 프로세스 인젝션
    ["NtOpenProcess", "NtAllocateVirtualMemory", "NtWriteVirtualMemory", "NtCreateThreadEx", "NtQuerySystemInformation"],  # 프로세스 인젝션 및 안티 디버깅
    ["CreateRemoteThread", "VirtualAllocEx", "WriteProcessMemory", "SetThreadContext", "NtQueryInformationProcess"],  # 원격 스레드 실행 및 안티 디버깅
    ["NtQuerySystemInformation", "NtQueryInformationProcess", "NtSetInformationThread", "NtLoadDriver"],  # 안티 디버깅 및 드라이버 로딩

    # 정상 프로그램 샘플
    ["CreateFile", "ReadFile", "WriteFile", "CloseHandle", "RegOpenKeyEx"],  # 파일 입출력 및 레지스트리 접근
    ["socket", "connect", "send", "recv", "closesocket", "HttpOpenRequest"],  # 네트워크 통신 및 웹 요청
    ["RegOpenKeyEx", "RegQueryValueEx", "RegCloseKey", "OpenProcess"],  # 레지스트리 접근 및 프로세스 열기
    ["OpenProcess", "GetModuleFileNameEx", "EnumProcesses", "CloseHandle", "FindWindow"],  # 시스템 모니터링 및 UI 조작
]

labels = [
    1, 1, 1, 1,  # 첫 4개는 악성코드
    0, 0, 0, 0,  # 다음 4개는 정상
    1, 1, 1, 1,  # 악성코드
    0, 0, 0,  # 정상 프로그램
    1, 1, 1, 1,  # 추가 악성코드
    0, 0, 0, 0, 0,  # 추가 정상 프로그램
    1, 1, 1, 1,  # 더 많은 악성코드
    0, 0, 0, 0  # 더 많은 정상 프로그램
]  # 1: 악성코드, 0: 정상 프로그램

# 2. 정수 인코딩 (Tokenization)
unique_words = set(word for seq in api_sequences for word in seq)
api_vocab = {word: idx + 1 for idx, word in enumerate(unique_words)}  # 1부터 시작
encoded_sequences = [[api_vocab[word] for word in seq] for seq in api_sequences]

# 3. 시퀀스 패딩 (고정 길이 변환)
MAX_LEN = 10  # 고정된 시퀀스 길이
padded_sequences = [seq + [0] * (MAX_LEN - len(seq)) for seq in encoded_sequences]

# 4. PyTorch Tensor 변환
X_train = torch.tensor(padded_sequences, dtype=torch.long)
y_train = torch.tensor(labels, dtype=torch.float32)

# 5. 데이터셋 및 DataLoader 생성
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------------------------------
# 모델 정의 및 학습
# --------------------------------------------------

# 6. 하이퍼파라미터 설정
EMBED_DIM = 64  # 임베딩 차원
NUM_HEADS = 8   # 멀티 헤드 어텐션 개수
NUM_LAYERS = 6  # Transformer Encoder 레이어 개수
FFN_HIDDEN = 128 # FFN 차원
EPOCHS = 100     # 학습 에폭 수
LR = 0.001       # 학습률
VOCAB_SIZE = len(api_vocab) + 1  # 토큰 개수 (0 포함)

# 7. Transformer Encoder 기반 악성코드 탐지 모델 정의
class MalwareDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_hidden, max_len):
        super(MalwareDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_hidden, batch_first=True)
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

# 8. 모델 생성
model = MalwareDetector(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_HIDDEN, MAX_LEN)

# 9. 손실 함수 및 최적화 함수 설정
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 10. 모델 훈련
for epoch in range(EPOCHS):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:  # 10 에폭마다 출력
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# --------------------------------------------------
# 모델 평가
# --------------------------------------------------

# 11. 일반 모델 평가
test_sequences = [
    ["CreateFile", "ReadFile", "WriteFile", "CloseHandle"],
    ["socket", "connect", "send", "recv", "closesocket"],
    ["RegOpenKeyEx", "RegQueryValueEx", "RegCloseKey"],
    ["SetWindowsHookEx", "GetAsyncKeyState", "SendInput"],
    ["AdjustTokenPrivileges", "LookupPrivilegeValue", "OpenProcessToken"],
    ["NtLoadDriver", "NtUnloadDriver", "NtOpenFile"],
]
true_labels = [0, 0, 0, 1, 1, 1]

test_X = torch.tensor(
    [[api_vocab.get(word, 0) for word in seq] + [0] * (MAX_LEN - len(seq)) for seq in test_sequences],
    dtype=torch.long
)

with torch.no_grad():
    predictions = model(test_X).squeeze().numpy()
predicted_labels = (predictions >= 0.5).astype(int)

print("모델 평가 결과")
print(classification_report(true_labels, predicted_labels, target_names=["정상", "악성"]))

# 12. k-fold 교차 검증
def k_fold_cross_validation(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    all_reports = []

    for train_index, test_index in kf.split(X):
        fold += 1
        print(f"Fold {fold}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dataset_train = TensorDataset(X_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

        model = MalwareDetector(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_HIDDEN, MAX_LEN)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # Training loop
        for epoch in range(EPOCHS):
            for batch_X, batch_y in dataloader_train:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation
        with torch.no_grad():
            predictions = model(X_test).squeeze().numpy()
        predicted_labels = (predictions >= 0.5).astype(int)
        report = classification_report(y_test.numpy(), predicted_labels, target_names=["정상", "악성"], output_dict=True, zero_division=0)
        all_reports.append(report)

        # 폴드 결과 출력
        print(f"Fold {fold} Report:")
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
        print()

    # 평균 성능 계산
    avg_report = {
        "정상": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "악성": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "accuracy": 0,
        "macro avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "weighted avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
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

    print("Average Report:\n")
    for key, metrics in avg_report.items():
        if key == "accuracy":
            print(f"{key}: {metrics:.4f}\n")
        else:
            print(f"{key}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            print()

# k-fold 교차 검증 실행
k_fold_cross_validation(X_train, y_train)