import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from model import UNet


class DummyImageDataset(Dataset):
    def __init__(self, clean_data, noise_data):
        self.clean_data = torch.tensor(clean_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.noise_data = torch.tensor(noise_data, dtype=torch.float32).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        clean = self.clean_data[idx]
        noisy = self.noise_data[idx]
        return [clean, noisy]

def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return data

def add_noise(img, noise_type="gaussian", std=0.05):
    if noise_type == "gaussian":
        noisy = img + std * np.random.randn(*img.shape)
        return np.clip(noisy, 0, 1)
    return img


def train(x_train, x_train_noise, x_val, x_val_noise):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    dataset = DummyImageDataset(x_train, x_train_noise)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    dataset = DummyImageDataset(x_val, x_val_noise)
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for clean, noise in train_loader:
            clean = clean.to(device)
            noise = noise.to(device)
            optimizer.zero_grad()
            output = model(noise)

            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for clean, noise in val_loader:
                clean, noise = clean.to(device), noise.to(device)
                output = model(noise)

                loss = criterion(output, clean)
                val_loss += loss.item()

        temp_val_loss = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.5f}, Val Loss: {temp_val_loss:.5f}, Current LR:{scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

        if temp_val_loss < best_loss:
            torch.save(model.state_dict(), f"../AI-Medical-Imaging-Recon/src/checkpoint/model_epoch_{epoch+1}.pth")
            best_loss = temp_val_loss

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    labels = ['PNEUMONIA', 'NORMAL']
    img_size = 150
    train = get_training_data('../AI-Medical-Imaging-Recon/data/chest_xray/train')
    val = get_training_data('../AI-Medical-Imaging-Recon/data/chest_xray/val')

    x_train = []
    y_train = []

    x_val = []
    y_val = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    x_train = x_train.reshape(-1, img_size, img_size, 1)
    x_train_noise = [add_noise(img) for img in x_train]
    y_train = np.array(y_train)

    x_val = x_val.reshape(-1, img_size, img_size, 1)
    x_val_noise = [add_noise(img) for img in x_val]
    y_val = np.array(y_val)

    train(x_train, x_train_noise, x_val, x_val_noise)
