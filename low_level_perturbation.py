from torchvision.transforms import functional as TF
import random
from torch.nn.functional import cosine_similarity
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import os
from vision_models import FeatureExtractorFactory
import gc
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
# ==== 加载图片 ====
def load_images(folder):
    images = []
    filenames = sorted(os.listdir(folder))  # 按名字排序以保持顺序
    for fname in tqdm(filenames, desc="Loading images"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("L")
            images.append(np.array(img).flatten())  # 展平成一维
    return np.array(images)

def load_images_paths(folder):
    images = []
    filenames = sorted(os.listdir(folder))  # 按名字排序以保持顺序
    for fname in tqdm(filenames, desc="Loading images"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, fname)
            images.append(img_path)
    return images

def load_images_paths_with_angle(folder):
    images = []
    angles = []
    filenames = sorted(os.listdir(folder))  # 按名字排序以保持顺序
    for fname in tqdm(filenames, desc="Loading images"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, fname)
            images.append(img_path)
            # 假设文件名格式为 "name_angle.jpg"
            angle = int(fname.split('_')[1].split('.')[0])  # 提取角度
            angles.append(angle)
    return images, angles

def apply_low_level_perturbation(image):
    angle = random.uniform(-5, 5)  # 小角度旋转
    image = TF.rotate(image, angle)

    if random.random() > 0.5:
        image = TF.hflip(image)

    return image


def measure_feature_change(model, image_paths, perturb_fn, path=True):
    original_images = []
    perturbed_images = []

    for path_or_img in image_paths:
        image = Image.open(path_or_img).convert("RGB") if path else path_or_img
        perturbed = perturb_fn(image)

        original_images.append(image)
        perturbed_images.append(perturbed)

    original_feats = model.extract_features(original_images, path=False)
    perturbed_feats = model.extract_features(perturbed_images, path=False)

    # np.ndarray -> torch.Tensor
    original_feats = torch.tensor(original_feats)
    perturbed_feats = torch.tensor(perturbed_feats)
    
    # cosine similarity between corresponding features
    similarities = cosine_similarity(original_feats, perturbed_feats, dim=1)
    return similarities.mean().item()


def get_args():
    parser = argparse.ArgumentParser(description="Quantitize reducted features with LLE")
    parser.add_argument("--folder_path", required=True, help="Path to the folder containing images", action='append', nargs='+')
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the visualization")
    parser.add_argument("--model_name", help="Vision models", action='append', nargs='+')
    # parser.add_argument("--embedding", type=str, default="LLE", help="Dimensionality reduction method")
    # parser.add_argument("--scaler", type=str, default="None", help="Scaler to use for feature scaling")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint")
    # parser.add_argument("--metrics", help="Metrics to use for evaluation", action='append', nargs='+')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    folder_paths = args.folder_path[0] if args.folder_path else None
    for folder in folder_paths:
        # print(f"Folder path: {folder}")
        if not os.path.exists(folder):
            raise ValueError(f"Folder {folder} does not exist.")
        print(f"Will process folder: {folder}")
    
    model_names = args.model_name[0] if args.model_name else None
    if model_names is None:
        raise ValueError("Model names are required.")
    print(f"Model names: {model_names}")
    save_path = args.save_path
    checkpoint_path = args.checkpoint_path

    modelName_2_sim = {}
    for model_name in model_names:
    
        print(f"Processing model: {model_name}")
        model = FeatureExtractorFactory.create_feature_extractor(model_name, checkpoint_path=checkpoint_path)
        

        for folder in folder_paths:

            images = load_images_paths(folder)

            similarity = measure_feature_change(model, images, apply_low_level_perturbation, path=True)

            if model_name not in modelName_2_sim.keys():
                modelName_2_sim[model_name] = []

            modelName_2_sim[model_name].append(similarity)

        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save the results  

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    model_names = list(modelName_2_sim.keys())
    similarities = np.mean([modelName_2_sim[model_name] for model_name in model_names], axis=1)

    if save_path:
        np.save(os.path.join(save_path, f"perturbation_sensitivity_{date_time}.npy"), similarities)

    for model_name, similarity in modelName_2_sim.items():
        print(f"Model: {model_name}, Similarity: {similarity}")



    plt.figure(figsize=(10, 6))
    plt.plot(model_names, similarities, marker='o', linestyle='-', linewidth=2,)
    plt.xticks(rotation=45)
    plt.xlabel("Model / Layer")
    plt.ylabel("Cosine Similarity (Original vs Perturbed)")
    plt.title("Perturbation Sensitivity Across Models and Layers")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        save_path = os.path.join(save_path, f"perturbation_sensitivity_{date_time}.png")
        plt.savefig(save_path, dpi=400)
        print(f"Saved plot to {save_path}")
    else:
        print("No save path provided. Showing plot instead.")
        plt.show()



