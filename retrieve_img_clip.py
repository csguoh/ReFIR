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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    features_hr, paths_hr = load_features()

    results = {}
    for lr_image_name in sorted(os.listdir(lr_folder_path)):
        if lr_image_name.endswith('.png'): 
            lr_image_path = os.path.join(lr_folder_path, lr_image_name)
            lr_image = prepare_image(lr_image_path, preprocess).to(device)
            lr_features = extract_features(model, lr_image)

            best_match_paths = find_most_similar(lr_features.cpu().numpy(), features_hr, paths_hr)
            results[lr_image_name] = best_match_paths 

    return results

if __name__ == '__main__':
    matching_results = main('/home/tiger/gh/dataset/RealPhoto60')
    for lr_img, hr_img in matching_results.items():
        print(f"Low-res image {lr_img} is best matched with high-res image {hr_img}")

    with open('/home/tiger/gh/dataset/retrieve_realPhoto.json', 'w') as fp:
        json.dump(matching_results, fp, indent=4)

