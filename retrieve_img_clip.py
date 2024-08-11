import numpy as np
import os
from PIL import Image
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_features():
    features = np.load('/home/tiger/gh/dataset/div_feat.npy')
    file_paths = np.load('/home/tiger/gh/dataset/div_path.npy', allow_pickle=True)
    return features, file_paths

def prepare_image(image_path, preprocess):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    return image

def extract_features(model, image_tensor):
    model.eval()
    with torch.no_grad():
        features = model.encode_image(image_tensor)
    return features

def find_most_similar(features, all_features, all_file_paths):
    similarity_scores = cosine_similarity(features, all_features)
    most_similar_idx = np.argmax(similarity_scores, axis=1)
    return [all_file_paths[idx] for idx in most_similar_idx]

def main(lr_folder_path):
    # 加载CLIP模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    # 加载预计算的特征和相应的路径
    features_hr, paths_hr = load_features()

    # 处理文件夹中的每个低分辨率图像
    results = {}
    for lr_image_name in sorted(os.listdir(lr_folder_path)):
        if lr_image_name.endswith('.png'):  # 过滤png图像
            lr_image_path = os.path.join(lr_folder_path, lr_image_name)
            lr_image = prepare_image(lr_image_path, preprocess).to(device)
            lr_features = extract_features(model, lr_image)

            # 找到最相似的高分辨率图像
            best_match_paths = find_most_similar(lr_features.cpu().numpy(), features_hr, paths_hr)
            results[lr_image_name] = best_match_paths  # 已经按照顺序排好了，方便TopK索引

    return results

if __name__ == '__main__':
    # 假设 `path_to_lr_folder` 是低分辨率图像文件夹的路径
    matching_results = main('/home/tiger/gh/dataset/RealPhoto60')
    for lr_img, hr_img in matching_results.items():
        print(f"Low-res image {lr_img} is best matched with high-res image {hr_img}")

    # 将结果保存到JSON文件
    with open('/home/tiger/gh/dataset/retrieve_realPhoto.json', 'w') as fp:
        json.dump(matching_results, fp, indent=4)

    # 加载JSON到字典
    # with open('/home/tiger/gh/dataset/retrieve_realPhoto.json', 'r') as fp:
    #     data_loaded = json.load(fp)
