import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm
from vision_models import FeatureExtractorFactory
from dim_reductor import DimensionalityReducer

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


def visualize(features, save_path=None, model_name=""):
    # 这里假设 features 是一个 N x D 的数组，N 是图片数量，D 是每张图片的特征维度
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=np.arange(len(features)), cmap='hsv')
    plt.title(f"LLE of {model_name} Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.axis("equal")
    plt.colorbar(label="Image Index (Rotation Angle or Order)")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def get_args():
    parser = argparse.ArgumentParser(description="Visualize features using LLE")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the visualization")
    parser.add_argument("--model_name", type=str, default="", help="Name of the model used for feature extraction")
    parser.add_argument("--scaler", type=str, default="None", help="Scaler to use for feature scaling")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    folder_path = args.folder_path
    save_path = args.save_path
    model_name = args.model_name
    scaler = args.scaler
    checkpoint_path = args.checkpoint_path

    # 加载图片并展平
    images = load_images_paths(folder_path)  
    # 加载模型
    model = FeatureExtractorFactory.create_feature_extractor(model_name=model_name, checkpoint_path=checkpoint_path)
    # 提取特征
    features = model.extract_features(images, path=True)  
    print(f"Extracted features shape: {features.shape}")
    # 特征降维
    dim_reducer = DimensionalityReducer(n_components=2, scaler=scaler, embedding='LocallyLinearEmbedding', n_neighbors=10)
    features = dim_reducer.fit_transform(features)

    # 可视化
    visualize(features, save_path=save_path, model_name=model_name)

    print(f"Visualization saved to {save_path}" if save_path else "Visualization displayed.")