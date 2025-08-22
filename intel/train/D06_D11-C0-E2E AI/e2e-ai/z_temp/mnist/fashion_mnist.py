import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder

# 데이터 로딩 및 정규화
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_oh = encoder.transform(y_test.reshape(-1, 1))

# 하이퍼파라미터
input_size = 784
hidden_size = 128
output_size = 10
epochs = 10
lr = 0.1
batch_size = 64

# 가중치 초기화
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
b2 = np.zeros((1, output_size))

# 활성화 함수 및 미분
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return x > 0

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_pred.shape[0]

# 정확도 계산
def accuracy(preds, labels):
    return np.mean(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))

# 훈련 루프
for epoch in range(epochs):
    permutation = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[permutation]
    y_train_shuffled = y_train_oh[permutation]

    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        # Forward pass
        z1 = np.dot(x_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)

        # Loss
        loss = cross_entropy(a2, y_batch)

        # Backpropagation
        dz2 = a2 - y_batch
        dW2 = np.dot(a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(x_batch.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        # 가중치 업데이트
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # 에포크마다 정확도 출력
    test_z1 = np.dot(x_test, W1) + b1
    test_a1 = relu(test_z1)
    test_z2 = np.dot(test_a1, W2) + b2
    test_a2 = softmax(test_z2)

    acc = accuracy(test_a2, y_test_oh)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Test Accuracy: {acc:.4f}")

# ============================================
# ✅ 테스트 샘플 10개에 대해 추론 결과 출력
# ============================================
print("\n🧪 Test Samples Inference Results:")
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

num_samples = 10
x_sample = x_test[:num_samples]
y_true = y_test[:num_samples]

# 추론
z1 = np.dot(x_sample, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)
y_pred = np.argmax(a2, axis=1)

# 출력
for i in range(num_samples):
    print(f"Sample {i+1}:")
    print(f"  👕 True Label   : {y_true[i]} ({class_names[y_true[i]]})")
    print(f"  🤖 Predicted    : {y_pred[i]} ({class_names[y_pred[i]]})")
    print("-" * 40)

