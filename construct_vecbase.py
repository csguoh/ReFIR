import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np



class RetrivalDataset(Dataset):
    def __init__(self, folder_path):
        self.file_list = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_path

def extract_features(dataset_path, batch_size=32):
    # 加载预训练模型
    model = models.vgg16(pretrained=True)
    model = model.features  # 使用卷积层的输出作为特征
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据
    dataset = RetrivalDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 提取特征
    features = []
    file_paths = []
    with torch.no_grad():
        for data in dataloader:
            images, paths = data
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            output = model(images)
            output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
            features.append(output.squeeze().cpu().numpy())
            file_paths.extend(paths)

    features = np.vstack(features)
    return features, file_paths



if __name__ == '__main__':
    # 使用函数
    features, file_paths = extract_features('/home/tiger/gh/dataset/DIV2K/DIV2K_train_HR')
    np.save('/home/tiger/gh/dataset/div_feat.npy', features)
    np.save('/home/tiger/gh/dataset/div_path.npy', file_paths)
