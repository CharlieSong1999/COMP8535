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

# from sklearn.linear_model import Ridge
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.decomposition import PCA
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.neural_network import MLPRegressor

# def evaluate_angle_predictability(features, angles):
#     """
#     评估特征是否可以线性预测相机旋转角度。
    
#     Parameters:
#     - features: (N, D) numpy array，提取的图像特征
#     - angles: (N,) numpy array，对应每张图像的旋转角度（单位：度，范围 [0, 360)）
#     - alpha: Ridge 回归的正则项
#     - cv: 交叉验证折数
#     - scale: 是否标准化特征

#     Returns:
#     - 平均 R² 分数（越高表示特征越清晰地编码了视角信息）
#     """

#     # 自动设置 PCA 维度小于样本数量
#     max_components = min(features.shape[0] - 1, features.shape[1])  # N-1 vs D
#     pca = PCA(n_components=min(32, max_components))  # 或更保守用 8

#     # y = np.stack([np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles))], axis=1)

#     pipeline = make_pipeline(
#         StandardScaler(),
#         pca,
#         MLPRegressor(hidden_layer_sizes=(64,), activation='relu', alpha=1e-3, max_iter=10000)
#     )

#     scores = cross_val_score(pipeline, features, angles, cv=5, scoring='r2')
#     return scores.mean()

# import numpy as np
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPRegressor
# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.metrics import make_scorer
# import math
# from sklearn.decomposition import PCA


# class SinCosAngleWrapper(BaseEstimator, RegressorMixin):
#     """
#     A wrapper for any regressor that uses (sin(angle), cos(angle)) as targets
#     and outputs angles in degrees.
#     """
#     def __init__(self, base_regressor=None):
#         self.base_regressor = base_regressor if base_regressor else MLPRegressor(
#             hidden_layer_sizes=(64, 64),
#             activation='relu',
#             alpha=1e-3,
#             max_iter=10000,
#             random_state=42
#         )

#     def fit(self, X, y_deg):
#         y_rad = np.deg2rad(y_deg)
#         y_sin_cos = np.stack([np.sin(y_rad), np.cos(y_rad)], axis=1)
#         self.base_regressor.fit(X, y_sin_cos)
#         return self

#     def predict(self, X):
#         y_sin_cos_pred = self.base_regressor.predict(X)
#         return np.rad2deg(np.arctan2(y_sin_cos_pred[:, 0], y_sin_cos_pred[:, 1])) % 360


# def mean_angular_error(y_true_deg, y_pred_deg):
#     """Mean angular error in degrees."""
#     delta = (y_pred_deg - y_true_deg + 180) % 360 - 180
#     return np.mean(np.abs(delta))


# # Make it usable with cross_val_score
# angular_error_scorer = make_scorer(mean_angular_error, greater_is_better=False)

# def evaluate_model_with_angle_regression(X, angles_deg, use_mlp=True):
#     """
#     Full pipeline to evaluate angle predictability using sin-cos and multi-layer MLP.
    
#     Parameters:
#         X: np.ndarray of shape (N, D)
#         angles_deg: np.ndarray of shape (N,)
#         use_mlp: whether to use MLP or fallback to Ridge
    
#     Returns:
#         mean angular error (lower is better)
#     """
#     model = SinCosAngleWrapper()
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('pca', PCA(n_components=min(16, X.shape[1]))),  # Adjust PCA components
#         ('regressor', model)
#     ])

#     X = X - X[0] # Difference from the first image whose angle is 0

#     scores = cross_val_score(pipeline, X, angles_deg, cv=5, scoring=angular_error_scorer)
#     return -scores.mean()  # Return positive error


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

# ========== 数据集类 ==========
class PoseDataset(Dataset):
    def __init__(self, features, angles):
        """
        features: (N, D) or (N, H, W, D)
        angles: (N,) in degrees
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        theta_rad = np.deg2rad(angles)
        self.targets = torch.tensor(np.stack([np.sin(theta_rad), np.cos(theta_rad)], axis=1), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# ========== CNN 特征聚合器 + 回归器 ==========
class CNNPoseRegressor(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (N, H, W, D) -> (N, D, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        out = self.regressor(x)
        return out

class MLPPoseRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # sin, cos
        )

    def forward(self, x):
        return self.model(x)

# ========== 角度误差计算 ==========
def mean_angular_error(pred, target):
    """pred, target: (N, 2), representing (sin, cos)"""
    pred_angle = torch.atan2(pred[:, 0], pred[:, 1])
    target_angle = torch.atan2(target[:, 0], target[:, 1])
    diff = torch.rad2deg((pred_angle - target_angle + np.pi) % (2 * np.pi) - np.pi)
    return diff.abs().mean().item()

# ========== 评估主函数 ==========
def evaluate_pose_regression(features, angles, epochs=50, batch_size=16, lr=1e-3, folds=5):
    features = np.array(features)
    angles = np.array(angles)
    if features.ndim == 4:  # (N, H, W, D) keep it
        pass
    elif features.ndim == 3:  # (N, H*W, D) → reshape to (N, H, W, D)
        N, HW, D = features.shape
        H = W = int(HW ** 0.5)
        features = features.reshape(N, H, W, D)
    elif features.ndim == 2:  # (N, D) 
        flat_input = True
    else:
        raise ValueError("Unsupported feature shape")

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    errors = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for train_idx, val_idx in kf.split(features):
        train_set = PoseDataset(features[train_idx], angles[train_idx])
        val_set = PoseDataset(features[val_idx], angles[val_idx])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # 初始化模型
        if features.ndim == 2:
            model = MLPPoseRegressor(input_dim=features.shape[1]).to(device)
        else:
            model = CNNPoseRegressor(in_channels=features.shape[-1]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Pose Regression Training"):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 验证
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                pred = model(x_batch).cpu()
                preds.append(pred)
                targets.append(y_batch)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        error = mean_angular_error(preds, targets)
        errors.append(error)

    return np.mean(errors)



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

    modelName_2_score = {}
    for model_name in model_names:
    
        print(f"Processing model: {model_name}")
        model = FeatureExtractorFactory.create_feature_extractor(model_name, checkpoint_path=checkpoint_path)
        

        for folder in folder_paths:

            images, angles = load_images_paths_with_angle(folder)

            features = model.extract_features(images, path=True, flatten=False)

            print(f"Extracted features shape: {features.shape}")

            score = evaluate_pose_regression(features, angles)

            if model_name not in modelName_2_score.keys():
                modelName_2_score[model_name] = []

            modelName_2_score[model_name].append(score)
        
        print(f"Model: {model_name}, Angle_predictability: {modelName_2_score[model_name]}")

        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save the results  

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    for model_name, similarity in modelName_2_score.items():
        print(f"Model: {model_name}, Angle_difference: {similarity}")

    model_names = list(modelName_2_score.keys())
    similarities = [modelName_2_score[model_name][0] for model_name in model_names]

    if save_path:
        np.save(os.path.join(save_path, f"angle_pred_{date_time}.npy"), similarities)

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, similarities)
    plt.xticks(rotation=45)
    plt.xlabel("Model / Layer")
    plt.ylabel("Angle Predictability (Error in degrees)")
    plt.title("Angle Predictability of Different Models")
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        save_path = os.path.join(save_path, f"angle_pred_{date_time}.png")
        plt.savefig(save_path, dpi=400)
        print(f"Saved plot to {save_path}")
    else:
        print("No save path provided. Showing plot instead.")
        plt.show()



