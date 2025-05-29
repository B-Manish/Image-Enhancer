import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

# ✅ SRCNN Model Definition
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)  # Feature extraction
        self.relu1 = nn.ReLU()
        
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)  # Pattern mapping
        self.relu2 = nn.ReLU()
        
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)   # Image reconstruction

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x

# ✅ Preprocess the input image
def preprocess_image(image_path):
    hr = cv2.imread(image_path)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    
    # Resize to 720p to simulate low-res
    lr = cv2.resize(hr, (850, 480), interpolation=cv2.INTER_CUBIC)
    # Resize back to 1080p to match HR size (blurry input)
    lr_upscaled = cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    hr_resized = cv2.resize(hr, (1920, 1080), interpolation=cv2.INTER_CUBIC)  # For training consistency

    # Convert to tensors
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_upscaled).unsqueeze(0)  # Add batch dimension
    hr_tensor = to_tensor(hr_resized).unsqueeze(0)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Upscaled Low-Res Image (Blurry)")
    plt.imshow(lr_upscaled)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("High-Res Image (Ground Truth)")
    plt.imshow(hr_resized)
    plt.axis('off')

    plt.tight_layout()
    plt.show()




        

    return lr_tensor, hr_tensor

# ✅ Save output tensor as image
def save_output(tensor, path):
    image = tensor.squeeze(0).cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255  # CHW → HWC
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)


def test():
    preprocess_image("frame_00000.jpg")    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    image_folder = "frames"
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    num_epochs = 1

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i,image_path in enumerate(image_paths):
            print(i)
            lr_image, hr_image = preprocess_image(image_path)
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)

            optimizer.zero_grad()
            output = model(lr_image)
            loss = criterion(output, hr_image)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")


    # Save model weights
    torch.save(model.state_dict(), "srcnn_final.pth")
    print("Model weights saved as srcnn_final.pth")


def test_model_on_image(model_path, test_image_path, output_path="enhanced_output.png"):
    # Load model
    model = SRCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Preprocess test image
    lr_tensor, _ = preprocess_image(test_image_path)

    # Run inference
    with torch.no_grad():
        output = model(lr_tensor)

    # Save result
    save_output(output, output_path)
    print(f"Enhanced image saved to {output_path}")    

if __name__ == "__main__":
    # main()
    # test()
    # test_model_on_image("srcnn_final.pth","test2.jpg")
    preprocess_image("frame_00000.jpg")


    # print(torch.cuda.is_available())        # Should be True
    # print(torch.cuda.device_count())        # Should be >= 1
    # print(torch.cuda.get_device_name(0))
