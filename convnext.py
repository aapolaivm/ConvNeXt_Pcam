import subprocess
import sys

# Asenna tarvittavat kirjastot, jos puuttuvat

def ensure_packages():
    packages = [
        "torch", "torchvision", "timm", "scikit-learn", "pandas", "numpy", "gdown", "tqdm"
    ]
    for package in packages:
        try:
            __import__(package if package != "scikit-learn" else "sklearn")
        except ImportError:
            print(f"\nAsennetaan puuttuva kirjasto: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

ensure_packages()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import PCAM
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# Laite
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformit
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Datan lataus
train_dataset = PCAM(root="./data", split='train', transform=transform, download=True)
val_dataset = PCAM(root="./data", split='val', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Mallin lataus
model = timm.create_model("convnext_large", pretrained=True, num_classes=2)
model.to(DEVICE)

# Optimointi ja tappio
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.001)

# Mallin tallennuspolku
os.makedirs("models", exist_ok=True)
model_path = "models/convnext_pcam_best.pth"

# Koulutus- ja validaatiolooppi
EPOCHS = 3
best_auc = 0.0

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Validaatio
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Todennäköisyys luokalle 1
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)
    print(f"Validation AUC: {auc:.4f}")

    # Tallennetaan paras malli
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), model_path)
        print(f"\n>> Uusi paras malli tallennettu: {model_path} (AUC: {auc:.4f})\n")
