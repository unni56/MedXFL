import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class NIHDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.labels_list = [
            'Atelectasis','Cardiomegaly','Effusion','Infiltration',
            'Mass','Nodule','Pneumonia','Pneumothorax',
            'Consolidation','Edema','Emphysema','Fibrosis',
            'Pleural_Thickening','Hernia'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image Index']
        labels = self.data.iloc[idx]['Finding Labels']

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label_vector = [0] * 14

        for label in labels.split('|'):
            if label in self.labels_list:
                label_vector[self.labels_list.index(label)] = 1

        label_vector = torch.tensor(label_vector, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_vector