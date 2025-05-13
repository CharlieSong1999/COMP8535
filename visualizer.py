import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm
from vision_models import FeatureExtractorFactory
from dim_reductor import DimensionalityReducer
import math
from datetime import datetime

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

def visualize_models(args, features_dict, save_path=None):
    """
    可视化多个模型降维后的二维特征（如 PCA 或 LLE）。
    
    :param args: argparse.Namespace, 命令行参数。
    :param features_dict: dict[str, np.ndarray], 每个键是模型名，每个值是形状为 (M, 2) 的降维特征。
    :param save_path: str or None，如果提供则保存图像，否则直接展示。
    """
    num_models = len(features_dict)
    cols = math.ceil(math.sqrt(num_models))
    rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

    for idx, (model_name, features) in enumerate(features_dict.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        scatter = ax.scatter(features[:, 0], features[:, 1], c=np.arange(len(features)), cmap='hsv')
        ax.set_title(f"{model_name}", fontsize=12)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.axis("equal")
        ax.grid(True)

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Image Index (Rotation Angle or Order)")

    # 移除多余的子图
    for idx in range(num_models, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    plt.tight_layout()

    if save_path:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        embedding = args.embedding
        save_path = os.path.join(save_path, f"{embedding}_visualization_{date_time}.png")
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def get_args():
    parser = argparse.ArgumentParser(description="Visualize features using LLE")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the visualization")
    parser.add_argument("--model_name", help="Vision models", action='append', nargs='+')
    parser.add_argument("--embedding", type=str, default="LLE", help="Dimensionality reduction method")
    parser.add_argument("--scaler", type=str, default="None", help="Scaler to use for feature scaling")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    folder_path = args.folder_path
    save_path = args.save_path
    model_names = args.model_name[0] if args.model_name else None
    if model_names is None:
        raise ValueError("Model name must be provided.")
    embedding = args.embedding
    scaler = args.scaler
    checkpoint_path = args.checkpoint_path

    # 加载图片并展平
    images = load_images_paths(folder_path)  

    features_dict = {}
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        # 加载模型
        model = FeatureExtractorFactory.create_feature_extractor(model_name=model_name, checkpoint_path=checkpoint_path)
        # 提取特征
        features = model.extract_features(images, path=True)  
        print(f"Extracted features shape: {features.shape}")
        # 特征降维
        dim_reducer = DimensionalityReducer(n_components=2, scaler=scaler, embedding=embedding, n_neighbors=10)
        features = dim_reducer.fit_transform(features)
        features_dict[model_name] = features

    # 可视化
    visualize_models(args, features_dict, save_path=save_path)

    print(f"Visualization saved to {save_path}" if save_path else "Visualization displayed.")