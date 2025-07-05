import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Using:", device)

# --- Define the model (must match training architecture) ---
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

# --- Load model ---
model = SimpleConvNet().to(device)
model.load_state_dict(torch.load("models/depixel_model.pt", map_location=device))
model.eval()

# --- Load test image ---
filename = "data/pixelated_test.jpg"
img = Image.open(filename).convert("RGB")

# --- Preprocess ---
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)  # Shape: (1, 3, H, W)

# --- Predict ---
with torch.no_grad():
    output_tensor = model(input_tensor)[0].cpu()  # Shape: (3, H, W)

# --- Convert tensors to NumPy for display
output_np = output_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
input_np = np.asarray(img.resize((512, 512))) / 255.0  # normalize to [0,1]

# --- Plot Input and Output
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_np)
plt.title("Input")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_np)
plt.title("Output")
plt.axis('off')

plt.tight_layout()
plt.show()

# --- Save output
output_img = (output_np * 255).astype(np.uint8)
Image.fromarray(output_img).save("gg.jpg")
print("✅ Depixelated image saved as gg.jpg")
