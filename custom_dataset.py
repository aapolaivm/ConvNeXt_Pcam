import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): Hakemisto, jossa kuvat ovat seuraavassa rakenteessa:
                - img_dir/0/kuva1.jpg, img_dir/0/kuva2.jpg, ... (luokka 0)
                - img_dir/1/kuva1.jpg, img_dir/1/kuva2.jpg, ... (luokka 1)
            transform (callable, optional): Kuvanmuunnostoiminnot
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # Etsi luokat (alikansiot)
        self.classes = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        self.class_to_idx = {c: int(c) for c in self.classes}  # Oletetaan että kansiot ovat nimeltään "0", "1" jne.
        
        self.img_paths = []
        self.labels = []
        
        # Kerää kuvat ja niiden luokat
        for class_name in self.classes:
            class_dir = os.path.join(img_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # Lataa kuva
        image = Image.open(img_path).convert('RGB')
        
        # Sovella transformaatioita
        if self.transform:
            image = self.transform(image)
            
        return image, label