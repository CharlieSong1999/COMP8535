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