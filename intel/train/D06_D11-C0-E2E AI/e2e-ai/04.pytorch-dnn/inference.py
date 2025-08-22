import os
import torch
from PIL import Image
from dnn import DNN
from utils import set_seed

def preprocess_image_tensor(img_path):
    img = Image.open(img_path).convert("L").resize((28, 28))
    img = torch.tensor(list(img.getdata()), dtype=torch.float32).view(-1)
    return img / 255.0

def load_images_in_folder(folder):
    data, filenames = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            data.append(preprocess_image_tensor(path))
            filenames.append(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return torch.stack(data), filenames if data else (torch.empty(0), filenames)

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("model_tensor.pth", map_location=device)
    label_map = checkpoint["label_map"]
    id_to_label = {v: k for k, v in label_map.items()}

    input_size = 28 * 28
    output_size = len(label_map)
    hidden_sizes = [512, 256, 128]

    model = DNN(input_size, hidden_sizes, output_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_root = "dataset/test"
    total_correct, total_samples = 0, 0

    print("Detailed Inference Results\n==========================")
    for class_name in sorted(os.listdir(test_root)):
        folder = os.path.join(test_root, class_name)
        if not os.path.isdir(folder): continue

        X, filenames = load_images_in_folder(folder)
        if X.numel() == 0:
            print(f"{class_name}: No images found.")
            continue

        X = X.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1).cpu()
        true_label = label_map[class_name]

        correct = 0
        print(f"\nClass: {class_name}\n" + "-"*40)
        for i, pred in enumerate(preds):
            pred_label = id_to_label[pred.item()]
            is_correct = pred.item() == true_label
            mark = "✔️" if is_correct else "❌"
            print(f"{filenames[i]:25} → Predicted: {pred_label:15} {mark}")
            if is_correct:
                correct += 1

        acc = correct / len(preds) * 100
        print(f"\n{class_name} Accuracy: {acc:.2f}% ({correct}/{len(preds)})")
        total_correct += correct
        total_samples += len(preds)

    print("\n==========================")
    print(f"Overall Accuracy : {total_correct / total_samples * 100:.2f}% ({total_correct}/{total_samples})")

if __name__ == "__main__":
    set_seed(42)
    inference()

