import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from vision_models import FeatureExtractorFactory
from dim_reductor import DimensionalityReducer
from metrics import MetricsFactory
from datetime import datetime
import torch
import gc
import pandas as pd

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


def get_args():
    parser = argparse.ArgumentParser(description="Quantitize reducted features with LLE")
    parser.add_argument("--folder_path", required=True, help="Path to the folder containing images", action='append', nargs='+')
    # parser.add_argument("--save_path", type=str, default=None, help="Path to save the visualization")
    parser.add_argument("--model_name", help="Vision models", action='append', nargs='+')
    parser.add_argument("--embedding", type=str, default="LLE", help="Dimensionality reduction method")
    parser.add_argument("--scaler", type=str, default="None", help="Scaler to use for feature scaling")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--metrics", help="Metrics to use for evaluation", action='append', nargs='+')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    folder_paths = args.folder_path[0] if args.folder_path else None
    for folder in folder_paths:
        # print(f"Folder path: {folder}")
        if not os.path.exists(folder):
            raise ValueError(f"Folder {folder} does not exist.")
        print(f"Will process folder: {folder}")
    # save_path = args.save_path
    model_names = args.model_name
    scaler = args.scaler
    checkpoint_path = args.checkpoint_path
    metric_names = args.metrics[0] if args.metrics else []
    model_names = model_names[0] if model_names else None
    if model_names is None:
        raise ValueError("Model name must be provided.")
    embedding = args.embedding
    print(f"Metrics: {metric_names}")
    print(f"Model names: {model_names}")
    # print(f"Folder path: {folder_paths}")

    

    Results = {}
    Records = []

    for folder_path in folder_paths:

        normalized = os.path.normpath(folder_path)           # Handles slashes on any OS
        parts = normalized.split(os.sep)              # Split on directory separator
        data_name = "_".join(part for part in parts if part != "..")  # Remove parent dir

        
        for model_name in model_names:
            print(f"Processing model: {model_name}")
            # 加载图片并展平
            images, angles = load_images_paths_with_angle(folder_path)  
            # 加载模型
            model = FeatureExtractorFactory.create_feature_extractor(model_name=model_name, checkpoint_path=checkpoint_path)
            # 提取特征
            features = model.extract_features(images, path=True)  
            print(f"Extracted features shape: {features.shape}")
            # 特征降维
            dim_reducer = DimensionalityReducer(n_components=2, scaler=scaler, embedding=embedding, n_neighbors=10)
            features = dim_reducer.fit_transform(features)

            # 可视化
            # angles to 2-dimensional array
            angles = np.array(angles)[:, None]
            if not 'angles' in Results:
                Results['angles'] = angles
            # print(f'Angles : {angles}')
            metrics = {}
            for matric_name in metric_names:
                # print(f"Computing metrics for {matric_name}...")
                metric = MetricsFactory.create_metrics(matric_name, data=angles, embedding=features, n_neighbors=5)
                metrics[matric_name] = metric.compute()
                # print(f"Metrics ({matric_name}): {metrics[matric_name]:.3f}")
            Results[model_name] = metrics

            a_record = {
                'Data': data_name,
                'Model': model_name,
                'Dimension reductor': embedding,
                'Metrics_spearman_correlation': metrics.get('spearman_correlation', None),
                'Metrics_pearson_correlation': metrics.get('pearson_correlation', None),
                'Metrics_trustworthiness': metrics.get('trustworthiness', None),
                'Metrics_continuity': metrics.get('continuity', None),
            }

            Records.append(a_record)

            del model
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save results to a file
    # Get date and time

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    df = pd.DataFrame.from_dict(Records)
    print(df)
    columns = ['Data', 'Model', 'Dimension reductor', 'Metrics_spearman_correlation', 'Metrics_pearson_correlation', 'Metrics_continuity', 'Metrics_trustworthiness']
    df.to_excel(f"../xlsx/results_{date_time}.xlsx", index=False, columns=columns)
    # print the results
    for model_name, metrics in Results.items():
        print(f"Model: {model_name}")
        if model_name == 'angles':
            print(f"Angles: {metrics}")
        else:
            for metric_name, metric in metrics.items():
                print(f"Metrics ({metric_name}): {metric:.3f}")
    # print(f"Metrics ({metric_name}): {metrics:.3f}")