import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.resnet import TBResNet
from pretraining.nih_dataset import NIHDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataset = NIHDataset(
    csv_file="data/nih/Data_Entry_2017.csv",
    img_dir="data/nih/images",
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = TBResNet(num_classes=14).to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


EPOCHS = 4

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "saved_models/resnet_pretrained.pth")

print("Pretraining done and model saved.")