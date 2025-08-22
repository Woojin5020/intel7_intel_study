import os
import torch
import random
from PIL import Image

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_image_tensor(img_path):
    img = Image.open(img_path).convert("L").resize((28, 28))
    img = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 28, 28)
    return img.view(-1) / 255.0

def one_hot_encode_tensor(label_index, num_classes):
    return torch.nn.functional.one_hot(torch.tensor(label_index), num_classes).float()

def load_dataset_tensor(base_path):
    X, y = [], []
    label_map = {}
    class_names = sorted(os.listdir(base_path))
    for idx, label_name in enumerate(class_names):
        label_map[label_name] = idx
        folder = os.path.join(base_path, label_name)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            try:
                x = preprocess_image_tensor(img_path)
                X.append(x)
                y.append(one_hot_encode_tensor(idx, len(class_names)))
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
    return torch.stack(X), torch.stack(y), label_map

