import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
import pickle

def load_test_dataset(root):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load class names and label map
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    label_map = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    # Load test data
    test_dataset = load_test_dataset("dataset/test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # Load model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("best_resnet50.pth", map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    print(classification_report(all_targets, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate()

