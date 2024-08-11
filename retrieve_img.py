import numpy as np
import os
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import json


def load_features():
    features = np.load('/home/tiger/gh/dataset/div_feat.npy')
    file_paths = np.load('/home/tiger/gh/dataset/div_path.npy', allow_pickle=True)
    return features, file_paths

def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def extract_features(model, image_tensor):
    model.eval()
    with torch.no_grad():
        features = model(image_tensor)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
    return features

def find_most_similar(features, all_features, all_file_paths):
    similarity_scores = cosine_similarity(features, all_features)
    most_similar_idx = np.argmax(similarity_scores, axis=1)
    return [all_file_paths[idx] for idx in most_similar_idx], similarity_scores.max()

def main(lr_folder_path):
    # Use VGG16 model's features for this example
    model = models.vgg16(pretrained=True).features
    # Load pre-computed features and corresponding paths
    features_hr, paths_hr = load_features()

    # Prepare the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Process each low-resolution image in the folder
    results = {}
    low_relevence_list =  []
    for lr_image_name in sorted(os.listdir(lr_folder_path)):
        if lr_image_name.endswith('.png'):  # filter for jpeg images
            lr_image_path = os.path.join(lr_folder_path, lr_image_name)
            lr_image = prepare_image(lr_image_path).to(device)
            lr_features = extract_features(model, lr_image)

            # Find the most similar high-resolution image
            best_match_paths,max_simi_score = find_most_similar(lr_features.cpu().numpy(), features_hr, paths_hr)
            print(f'{lr_image_name}==>score:{max_simi_score}')
            if max_simi_score < 0.6:
                low_relevence_list.append(lr_image_name)
            results[lr_image_name] = best_match_paths 
    print(low_relevence_list)
    return results





if __name__ == '__main__':
    # Assuming `path_to_lr_folder` is the path to the low-resolution images folder
    matching_results = main('/home/tiger/gh/dataset/RealPhoto60')
    for lr_img, hr_img in matching_results.items():
        print(f"Low-res image {lr_img} is best matched with high-res image {hr_img}")


    # Save results to JSON file
    with open('/home/tiger/gh/dataset/retrieve_realPhoto.json', 'w') as fp:
        json.dump(matching_results, fp, indent=4)

