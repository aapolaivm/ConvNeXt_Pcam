import subprocess
import sys
import os

def ensure_packages():
    packages = [
        "torch", "torchvision", "timm", "scikit-learn", "pandas", "numpy", 
        "matplotlib", "seaborn", "tqdm"
    ]
    for package in packages:
        try:
            __import__(package if package != "scikit-learn" else "sklearn")
        except ImportError:
            print(f"\nAsennetaan puuttuva kirjasto: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

ensure_packages()

import torch
import timm
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import torch.nn.functional as F
import random
from PIL import Image

# Voit käyttää PCAM datasettiä tai omaa datasettiäsi
from torchvision.datasets import PCAM

# Jos käytät omaa datasettiä, käytä alla olevaa luokkaa apuna 
# from custom_dataset import CustomImageDataset

# Konfiguraatio
model_path = "models/convnext_pcam_best.pth"  # Koulutetun mallin polku
results_dir = "results"                       # Hakemisto visualisoinneille
data_dir = "./data"                          # Datahakemisto
os.makedirs(results_dir, exist_ok=True)

# Laite
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Transformit - samat kuin koulutuksessa
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Tarkista onko data jo ladattu
data_exists = False
pcam_test_dir = data_dir + "/pcam/test"

if os.path.exists(pcam_test_dir):
    # Tarkista että hakemistossa on ainakin yksi kuvatiedosto
    for root, _, files in os.walk(pcam_test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                data_exists = True
                break
        if data_exists:
            break

# Datan lataus - käytä joko PCAM tai omaa datasettiä
if data_exists:
    print(f"Data löytyy jo hakemistosta {pcam_test_dir}, käytetään sitä.")
    # Käytä aikaisemmin ladattua dataa
    test_dataset = PCAM(root=data_dir, split='test', transform=transform, download=False)
else:
    print(f"Dataa ei löydy hakemistosta {pcam_test_dir}, ladataan data...")
    # Lataa data verkosta
    test_dataset = PCAM(root=data_dir, split='test', transform=transform, download=True)
    print("Data ladattu onnistuneesti!")

# Jos käytät omaa datasettiä, korjaa alla oleva rivi
#test_dataset = CustomImageDataset(img_dir="./data/oma_data/test", transform=transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Tarkista onko malli olemassa
if not os.path.exists(model_path):
    print(f"VIRHE: Mallia ei löydy polusta {model_path}")
    print("Varmista että olet kouluttanut mallin tai tarkista mallin polku.")
    sys.exit(1)

# Mallin lataus
print(f"Ladataan malli tiedostosta {model_path}...")
model = timm.create_model("convnext_large", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Malli ladattu onnistuneesti!")

# Testaus
all_labels = []
all_probs = []
all_preds = []
images_to_show = []
correct_examples = []
incorrect_examples = []

print(f"Evaluoidaan malli {len(test_dataset)} testikuvalla...")
with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating model")):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        
        # Tallenna todennäköisyydet ja todelliset luokat
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Ennusteiden tallennus
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        
        # Tallenna joitakin esimerkkikuvia visualisointia varten
        if len(images_to_show) < 20:
            for j in range(min(5, len(images))):
                if len(images_to_show) < 20:
                    img = images[j].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                    true_label = labels[j].item()
                    pred_label = predicted[j].item()
                    prob = probs[j, pred_label].item()
                    
                    if true_label == pred_label:
                        correct_examples.append((img, true_label, pred_label, prob))
                    else:
                        incorrect_examples.append((img, true_label, pred_label, prob))
                        
                    images_to_show.append((img, true_label, pred_label, prob))

print("Evaluointi valmis! Luodaan visualisointeja...")

# 1. ROC-käyrän visualisointi
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
plt.close()
print(f"ROC-käyrä tallennettu: {os.path.join(results_dir, 'roc_curve.png')}")

# 2. Confusionmatrix visualisointi
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Ennustettu luokka')
plt.ylabel('Todellinen luokka')
plt.title('Virheluokittelumatriisi (Confusion Matrix)')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()
print(f"Virheluokittelumatriisi tallennettu: {os.path.join(results_dir, 'confusion_matrix.png')}")

# 3. Luokitteluraportti
report = classification_report(all_labels, all_preds)
print("\nLuokitteluraportti:")
print(report)

# Tallenna raportti tekstitiedostoon
report_path = os.path.join(results_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"Luokitteluraportti tallennettu: {report_path}")

# 4. Visualisoi esimerkkikuvia ja niiden ennusteet
def visualize_predictions(examples, title, filename):
    if not examples:
        return
    
    n = min(len(examples), 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n):
        img, true_label, pred_label, prob = examples[i]
        
        # Normalisoi kuva näyttöä varten jos tarpeen
        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}\nProb: {prob:.2f}", color=color)
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    path = os.path.join(results_dir, filename)
    plt.savefig(path)
    plt.close()
    return path

# Visualisoi oikeat ja väärät ennusteet
correct_path = visualize_predictions(correct_examples, "Oikeat ennusteet", "correct_predictions.png")
incorrect_path = visualize_predictions(incorrect_examples, "Väärät ennusteet", "incorrect_predictions.png")

if correct_path:
    print(f"Oikeat ennusteet tallennettu: {correct_path}")
if incorrect_path:
    print(f"Väärät ennusteet tallennettu: {incorrect_path}")

# 5. Todennäköisyysjakauma
plt.figure(figsize=(10, 6))
sns.histplot(all_probs, bins=50, kde=True)
plt.xlabel("Luokan 1 todennäköisyys")
plt.ylabel("Frekvenssi")
plt.title("Todennäköisyysjakauma")
prob_path = os.path.join(results_dir, "probability_distribution.png")
plt.savefig(prob_path)
plt.close()
print(f"Todennäköisyysjakauma tallennettu: {prob_path}")

print(f"\nKaikki visualisoinnit tallennettu hakemistoon: {results_dir}")

# Näytä mallin suorituskykytiedot
correct = sum(1 for true, pred in zip(all_labels, all_preds) if true == pred)
total = len(all_labels)
accuracy = correct / total if total > 0 else 0

print("\nMallin suorituskyky:")
print(f"Tarkkuus (Accuracy): {accuracy:.4f} ({correct}/{total})")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Katso tarkemmat tiedot luokitteluraportista: {report_path}")