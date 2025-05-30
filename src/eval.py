import torch
import matplotlib.pyplot as plt
from model import UNet
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset


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

def visualize_sample(noisy, denoised, ground_truth):
    #noisy, denoised, ground_truth = noisy.cpu().squeeze().numpy(), denoised.cpu().squeeze().numpy(), ground_truth.cpu().squeeze().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(noisy, cmap="gray")
    plt.title("Noisy Input")

    plt.subplot(1, 3, 2)
    plt.imshow(denoised, cmap="gray")
    plt.title("Denoised Output")

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth, cmap="gray")
    plt.title("Ground Truth")

    plt.show()

if __name__ == "__main__":

    labels = ['PNEUMONIA', 'NORMAL']
    img_size = 150
    test = get_training_data('../AI-Medical-Imaging-Recon/data/chest_xray/test')
    x_test = []
    y_test = []

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    x_test = np.array(x_test) / 255
    x_test = x_test.reshape(-1, img_size, img_size, 1)
    x_test_noise = [add_noise(img) for img in x_test]
    y_test = np.array(y_test)

    dataset = DummyImageDataset(x_test, x_test_noise)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model_path = "./checkpoints/model_epoch_20.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    denoise_img = []
   
    for clean, noise in test_loader:
        noise = noise.to(device)
        with torch.no_grad():
            output_tensor = model(noise)
            output = output_tensor.squeeze().cpu().numpy()
            denoise_img.append(output)

    i = 10
    # Save result images
    os.makedirs("results", exist_ok=True)
    plt.imsave("results/noise image.png", x_test_noise[i].squeeze(), cmap="gray")
    plt.imsave("results/demoise image.png", denoise_img[0][i], cmap="gray")
    plt.imsave("results/groundtruth.png", x_test[i].squeeze(), cmap="gray")