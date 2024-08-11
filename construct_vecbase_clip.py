import torch
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

class RetrivalDataset(Dataset):
    def __init__(self, folder_path, preprocess):
        self.file_list = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]
        self.transform = preprocess

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path

def extract_features(dataset_path, batch_size=32):
    # Load the CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    # Prepare data
    dataset = RetrivalDataset(dataset_path, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Extract features
    features = []
    file_paths = []
    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            output = model.encode_image(images)
            features.append(output.cpu().numpy())
            file_paths.extend(paths)

    features = np.vstack(features)
    return features, file_paths

if __name__ == '__main__':
    # Use the function
    features, file_paths = extract_features('/home/tiger/gh/dataset/DF2K')
    np.save('/home/tiger/gh/dataset/div_feat.npy', features)
    np.save('/home/tiger/gh/dataset/div_path.npy', file_paths)
