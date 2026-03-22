import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.resnet import TBResNet
from pretraining.nih_dataset import NIHDataset


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= PATHS =================
CSV_PATH = "pretraining_data/Data_Entry_2017.csv"
IMG_DIR = "pretraining_data/images"
SAVE_PATH = "saved_models/resnet_pretrained.pth"

os.makedirs("saved_models", exist_ok=True)


# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


# ================= DATASET =================
dataset = NIHDataset(
    csv_file=CSV_PATH,
    img_dir=IMG_DIR,
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print("Dataset loaded. Total samples:", len(dataset))


# ================= MODEL =================
model = TBResNet(num_classes=14).to(device)


# ================= LOSS & OPTIMIZER =================
criterion = nn.BCEWithLogitsLoss()

# 🔥 Lower LR since resuming
optimizer = optim.Adam(model.parameters(), lr=5e-5)



START_EPOCH = 4  
TOTAL_EPOCHS = 10      

if os.path.exists(SAVE_PATH):
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    print("Loaded pretrained model. Resuming training from epoch 4...")
else:
    print("No checkpoint found. Starting fresh...")



best_loss = float("inf")

for epoch in range(START_EPOCH, TOTAL_EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]")

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

   
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print("💾 Best model saved!")


print("Training resumed and completed. Model saved at:", SAVE_PATH)