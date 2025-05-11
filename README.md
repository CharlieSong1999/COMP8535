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
python .\visualizer.py --folder_path desk --model_name same --checkpoint_path 'path/to/checkpoint_sam' 
```

lle with original images and metrics
```bash
python .\quantitizer.py --folder_path desk --model_name identity --scaler StandardScaler --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

lle with dinov2 features and metrics
```bash
python .\quantitizer.py --folder_path desk --model_name dinov2  --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

lle with stable-diffusion features and metrics
```bash
python .\quantitizer.py --folder_path desk --model_name stable_diffusion  --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

lle with clip features and metrics
```bash
python .\quantitizer.py --folder_path desk --model_name clip  --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

lle with deit features and metrics
```bash
python .\quantitizer.py --folder_path desk --model_name deit  --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

lle with sam features and metrics
```bash
python .\quantitizer.py --folder_path room --model_name identity  --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

Isomap with sam features and metrics
```bash
python .\quantitizer.py --embedding Isomap --folder_path ..\room2\render --model_name identity dinov2 stable_diffusion clip deit sam --metrics spearman_correlation pearson_correlation continuity trustworthiness
```

PCA with sam features and metrics
```bash
python .\quantitizer.py --embedding PCA --folder_path ..\Case\render ..\Desk_food\render ..\Excavator\render ..\Rhino\render ..\room2\render desk room --model_name identity dinov2 stable_diffusion clip deit sam --metrics spearman_correlation pearson_correlation continuity trustworthiness
```