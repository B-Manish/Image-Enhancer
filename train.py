import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # for progress bar

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Using:", device)

# --- Custom Dataset ---
class ImagePairDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.filenames = sorted(os.listdir(low_res_dir))
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low_path = os.path.join(self.low_res_dir, fname)
        high_path = os.path.join(self.high_res_dir, fname)

        if not os.path.exists(low_path) or not os.path.exists(high_path):
            raise FileNotFoundError(f"Missing file: {fname}")

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        x = self.transform(low_img)
        y = self.transform(high_img)
        return x, y

# --- Model ---
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- Paths ---
LOW_RES_PATH = "data/low_res"
HIGH_RES_PATH = "data/high_res"
MODEL_PATH = "models/depixel_model.pt"
os.makedirs("models", exist_ok=True)

# --- Dataset & DataLoader ---
batch_size = 8
dataset = ImagePairDataset(LOW_RES_PATH, HIGH_RES_PATH)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Training Setup ---
model = SimpleConvNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 2

# --- Training Loop with tqdm ---
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"📅 Epoch {epoch+1}/{epochs}", unit="batch")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} finished — Avg Loss: {avg_loss:.4f}")

# --- Save the Model ---
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
