import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from dnn import DNN
from utils import set_seed, load_dataset_tensor

def train_model(X_train, y_train, X_val, y_val, label_map, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_sizes = [512, 256, 128]

    model = DNN(input_size, hidden_sizes, output_size).to(device)

    labels = torch.argmax(y_train, dim=1)
    if args.use_class_weight:
        weights = compute_class_weight("balanced", classes=range(output_size), y=labels.cpu().numpy())
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    X_train = X_train.to(device)
    y_train = labels.to(device)
    X_val = X_val.to(device)
    y_val = torch.argmax(y_val, dim=1).to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_model = None
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(args.epoch):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        total_loss, correct = 0.0, 0

        for i in range(0, X_train.size(0), args.batch_size):
            idx = permutation[i:i+args.batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()

        train_loss = total_loss / X_train.size(0)
        train_acc = correct / X_train.size(0)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_loss < best_loss - args.delta:
            best_loss = val_loss
            best_model = model.state_dict()
            no_improve = 0
            print("★ Validation improved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping!")
                break

    torch.save({'model_state_dict': best_model, 'label_map': label_map}, 'model_tensor.pth')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.legend()
    plt.grid()
    plt.savefig("training_metrics_tensor.png")
    plt.close()

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    X, y, label_map = load_dataset_tensor("dataset/train")
    y_np = torch.argmax(y, dim=1).numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_ratio, stratify=y_np)
    train_model(X_train, y_train, X_val, y_val, label_map, args)

