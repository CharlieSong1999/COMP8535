 # Visualize reducted features

lle with original images:
```bash
python .\visualizer.py --folder_path desk --model_name identity --scaler StandardScaler
```

lle with dinov2:
```bash
python .\visualizer.py --folder_path desk --model_name dinov2 
```

lle with stable_diffusion 
```bash
python .\visualizer.py --folder_path desk --model_name stable_diffusion 
```

lle with clip
```bash
python .\visualizer.py --folder_path desk --model_name clip 
```

lle with deit
```bash
python .\visualizer.py --folder_path desk --model_name deit 
```

lle with sam (need to download the checkpoint from https://github.com/facebookresearch/segment-anything, vit-b)
```bash
python .\visualizer.py --folder_path desk --model_name sam --checkpoint_path 'path/to/checkpoint_sam' 
```

# Quantitize reducted features

## Supported arguments

- model_name
    - identity: original images
    - dinov2
        - dinov2-layer-0: image tokens from layer 0 of dinov2
    - stable_diffusion
    - clip
    - deit
    - sam

- metrics:
    - spearman_correlation
    - pearson_correlation
    - continuity
    - trustworthiness

- embedding
    - LLE
    - Isomap
    - TSNE
    - PCA
    - MDS

- folder_path: path to the data

## Example of commands

PCA with different vision models and metrics on different data
```bash
python .\quantitizer.py --embedding PCA --folder_path ..\Case\render ..\Desk_food\render ..\Excavator\render ..\Rhino\render ..\room2\render desk room --model_name identity dinov2 stable_diffusion clip deit sam --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

T-SNE with different vision models and metrics on different data
```bash
python .\quantitizer.py --embedding TSNE --folder_path ..\Case\render ..\Desk_food\render ..\Excavator\render ..\Rhino\render ..\room2\render desk room --model_name identity dinov2 stable_diffusion clip deit sam --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

PCA with different dinov2 with different layers and metrics on different data
```bash
python .\quantitizer.py --embedding PCA --folder_path ..\Case\render ..\Desk_food\render ..\Excavator\render ..\Rhino\render ..\room2\render desk room --model_name dinov2-layer-4 --metrics spearman_correlation pearson_correlation continuity trustworthiness
```