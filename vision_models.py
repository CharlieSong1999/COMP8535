import torch
from torchvision import transforms
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from PIL import Image
import cv2

import huggingface_hub
huggingface_hub.constants.HF_HUB_HTTP_TIMEOUT = 60

def is_pil_image(img):
    return isinstance(img, Image.Image)

def pil_to_cv2(pil_img):
    """
    Convert a PIL.Image to a cv2-style numpy array (BGR).
    """
    rgb = pil_img.convert("RGB")  # Ensure it's in RGB mode
    np_img = np.array(rgb)        # shape: (H, W, 3), RGB
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)  # Convert to BGR
    return cv_img

def ensure_cv2_format(image):
    """
    If image is a PIL.Image, convert to OpenCV format.
    Otherwise, return as is.
    """
    if isinstance(image, Image.Image):
        return pil_to_cv2(image)
    return image

class FeatureExtractor:

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def extract_features(self, images):
        """
        Extract features from the provided images using the loaded model.
        
        :param images: List of images to extract features from.
        :return: Extracted features.
        """
        # Placeholder for actual feature extraction logic
        # This should involve passing the images through the model and obtaining the features
        raise NotImplementedError("Subclasses should implement this method.")
    
class IdentityFeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__('identity', **kwargs)
        self.model = self.load_model()
    
    def load_model(self):
        """
        Load the identity model.
        
        :return: Identity model instance.
        """
        class IdentityModel(torch.nn.Module):
            def __init__(self):
                super(IdentityModel, self).__init__()

            def forward(self, x):
                return x
        
        return IdentityModel()
    
    def extract_features(self, images, path=True, flatten=True):
        """
        Extract features using the identity model.
        
        :param images: List of images to extract features from.
        :return: Extracted features (same as input).
        """
        features = []
        if path:
            for image in tqdm(images, desc="Loading images (identity)"):
                
                if flatten:
                    image = Image.open(image).convert("L").resize((512, 512))
                    features.append(np.array(image).flatten())  # 展平成一维
                else:
                    image = Image.open(image).resize((512, 512))
                    features.append(np.array(image))
        else:
            for image in tqdm(images, desc="Loading images (identity)"):
                if flatten:
                    features.append(image.flatten())
                else:
                    features.append(np.array(image, dtype=np.float32).reshape(-1, 3)) # for low-level perturbation
        
        features = np.array(features)
        # print(f"Feature shape: {features.shape}")

        return np.stack(features,axis=0)
    
class DINOv2FeatureExtractor(FeatureExtractor):

    def __init__(self, **kwargs):
        super().__init__('dinov2', **kwargs)
        self.model = self.load_model()

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']
            print(f"Using layer {self.layer_id} for DINOv2 feature extraction.")
        else:
            self.layer_id = None
        
        self.activations = {}

        if self.layer_id is not None:
            self.model.blocks[self.layer_id].register_forward_hook(self._get_hook(self.layer_id))
    
    def _get_hook(self, layer_id):
        def hook(module, input, output):
            self.activations[layer_id] = output.detach()
        return hook
    
    def load_model(self):
        """
        Load the DINOv2 model.
        
        :return: DINOv2 model instance.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').eval().cuda()
        return dinov2_model
        
    
    def extract_features(self, images, path=True, flatten=True):
        """
        Extract features using the DINOv2 model.
        
        :param images: List of images to extract features from.
        :return: Extracted features.
        """
        # Placeholder for actual feature extraction logic
        # This should involve passing the images through the DINOv2 model and obtaining the features

        # ===== 1. 预处理（注意尺寸是 518x518） =====
        transform = transforms.Compose([
            transforms.Resize((518, 518)),  # 必须匹配 large 模型
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        features = []
        for image in tqdm(images, desc="Extracting features with DINOv2"):
            # Apply the transformations to the image
            image = Image.open(image).convert("RGB") if path else image
            image_tensor = transform(image).unsqueeze(0).cuda() # (1, 3, 518, 518)
            # Pass the image through the model to get features 
            with torch.no_grad():
                feature = self.model(image_tensor) # (1, 1536)

            if self.layer_id is not None:
                feat = self.activations[self.layer_id] # (1, seq_len, dim)
                feat = feat[:, 1:, :]  # Remove CLS token (1, 1370, 1536) -> (1, 1369 (37*37), 1536)
                feat = feat.squeeze(0)  # Remove batch dimension (1, 1369, 1536) -> (1369, 1536)
                # print(f"Feature shape: {feat.shape}")
            else:
                feat = feature.squeeze(0)  # Remove batch dimension (1, 1536) -> (1536,)
            
            if flatten:
                feature = feat.flatten().cpu()
            else:
                if len(feat.shape) == 2:
                    feature = feat.unsqueeze(1)
                feature = feat 

            # print(f"Feature shape: {feature.shape}")
            
            features.append(feature.cpu())
        
        features = np.array(features)
        
        return features  # (N, 1536) or (N, 1369*1536) or (N, 1369, 1536)
        

class StableDiffusionFeatureExtractor(FeatureExtractor):

    def __init__(self, **kwargs):
        super().__init__('stable_diffusion', **kwargs)
        self.model = self.load_model()
    
    def load_model(self):
        """
        Load the Stable Diffusion model.
        
        :return: Stable Diffusion model instance.
        """
        # Placeholder for actual model loading logic
        # This should involve loading the Stable Diffusion model from a checkpoint or library
        # ==== 加载 Stable Diffusion v1.4 ====
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda"


        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        vae = pipe.vae.eval()  # 只用 VAE 的 encoder

        return vae
        
    
    def extract_features(self, images, path=True, flatten=True):
        """
        Extract features using the Stable Diffusion model.
        
        :param images: List of images to extract features from.
        :return: Extracted features.
        """
        # Placeholder for actual feature extraction logic
        # This should involve passing the images through the Stable Diffusion model and obtaining the features
        
        # ==== 图像预处理 ====
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        features = []
        for image in tqdm(images, desc="Extracting features with Stable Diffusion"):
            # Apply the transformations to the image
            image = Image.open(image).convert("RGB") if path else image
            image_tensor = preprocess(image).unsqueeze(0).to("cuda", dtype=torch.float16) # (1, 3, 512, 512)
            # Pass the image through the model to get features
            with torch.no_grad():
                latents = self.model.encode(image_tensor).latent_dist.sample() # (1, 4, 64, 64)
                latents = 0.18215 * latents # scale per SD paper
                # pooled = torch.nn.functional.adaptive_avg_pool2d(latents, (1, 1)) # (1, 4, 1, 1)

            if flatten:
                features.append(latents.flatten().cpu().numpy()) # (4,)
            else:
                features.append(latents.permute(0, 2, 3, 1).reshape(-1, 4).cpu().numpy()) # (64*64, 4)
        return np.stack(features, axis=0)
    

# class CLIPFeatureExtractor(FeatureExtractor):

#     def __init__(self, **kwargs):
#         super().__init__('clip', **kwargs)
#         self.model = self.load_model()

#     def load_model(self):
#         """
#         Load the CLIP model.
        
#         :return: CLIP model instance.
#         """
#         # Placeholder for actual model loading logic
#         # This should involve loading the CLIP model from a checkpoint or library
#         # ==== 加载 CLIP 模型 ====
#         # ==== 1. 加载模型 ====
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

#         return model
        
#     def extract_features(self, images, path=True, flatten=True):
#         """
#         Extract features using the CLIP model.
        
#         :param images: List of images to extract features from.
#         :return: Extracted features.
#         """
#         # Placeholder for actual feature extraction logic
#         # This should involve passing the images through the CLIP model and obtaining the features
        
#         # ==== 2. 图像预处理 ====
#         processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
#         features = []
#         for image in tqdm(images, desc="Extracting features with CLIP"):
#             # Apply the transformations to the image
#             image = Image.open(image).convert("RGB") if path else image
#             inputs = processor(images=image, return_tensors="pt").to("cuda")
#             # Pass the image through the model to get features
#             with torch.no_grad():
#                 outputs = self.model.get_image_features(**inputs)
#             features.append(outputs.cpu().numpy())
#         return np.stack(features, axis=0).squeeze(1)  # (N, 512)

class CLIPFeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__('clip', **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # ← 提前定义 device
        self.vit_layer = kwargs.get("vit_layer", -1)
        self.model = self.load_model()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def load_model(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.intermediate_feature = {}

        if self.vit_layer >= 0:
            max_layer = len(model.vision_model.encoder.layers)
            if self.vit_layer >= max_layer:
                raise ValueError(f"vit_layer={self.vit_layer} exceeds max layer {max_layer-1}")
            # 注册 hook
            model.vision_model.encoder.layers[self.vit_layer].register_forward_hook(self._get_hook())

        return model

    def _get_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # 提取主要输出 (B, seq_len, D)
            self.intermediate_feature['feature'] = output
        return hook_fn

    def extract_features(self, images, path=True, flatten=True):
        features = []

        for image in tqdm(images, desc="Extracting features with CLIP"):
            image = Image.open(image).convert("RGB") if path else image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                if self.vit_layer >= 0:
                    # trigger vision_model forward to run hook
                    _ = self.model.vision_model(**inputs)
                    feat = self.intermediate_feature['feature'].squeeze(0).cpu()  # (seq_len, D)
                else:
                    feat = self.model.get_image_features(**inputs).squeeze(0).cpu()  # (D,)

            if flatten:
                features.append(feat.flatten().numpy())
            else:
                features.append(feat.numpy())  # shape: (L, D)

        return np.array(features)
    

# class DeitFeatureExtractor(FeatureExtractor):

#     def __init__(self, **kwargs):
#         super().__init__('deit', **kwargs)
#         self.model = self.load_model()
    
#     def load_model(self):
#         """
#         Load the Deit model.
        
#         :return: Deit model instance.
#         """
#         # Placeholder for actual model loading logic
#         # This should involve loading the Deit model from a checkpoint or library
#         # ==== 设备 ====
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # ==== 加载 DeiT 模型（ViT-B/16） ====
#         self.model_name = "facebook/deit-base-distilled-patch16-224"
#         model = AutoModel.from_pretrained(self.model_name).to(device)
#         model.eval()
#         return model

#     def extract_features(self, images, path=True, flatten=True):
#         """
#         Extract features using the Deit model.
        
#         :param images: List of images to extract features from.
#         :return: Extracted features.
#         """
#         # Placeholder for actual feature extraction logic
#         # This should involve passing the images through the Deit model and obtaining the features
        
#         # ==== 1. 图像预处理 ====
#         processor = AutoImageProcessor.from_pretrained(self.model_name)

        
#         features = []
#         for image in tqdm(images, desc="Extracting features with DeiT"):
#             # Apply the transformations to the image
#             image = Image.open(image).convert("RGB") if path else image
#             inputs = processor(images=image, return_tensors="pt").to("cuda")
#             # Pass the image through the model to get features
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 if hasattr(outputs, "last_hidden_state"):
#                     if flatten:
#                         feat = outputs.last_hidden_state[:, :, :].flatten().cpu().numpy()  #other than CLS token
#                     else:
#                         feat = outputs.last_hidden_state[:, 1:-1, :].cpu().numpy() # (1, 196, 768) 0: CLS token, -1: Distillation token
#                 else:
#                     raise ValueError("模型输出格式不正确")
            
#             features.append(feat.squeeze())
#         return np.stack(features, axis=0)

class DeitFeatureExtractor(FeatureExtractor):

    def __init__(self, **kwargs):
        super().__init__('deit', **kwargs)
        self.vit_layer = kwargs.get("vit_layer", -1)
        self.model = self.load_model()
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "facebook/deit-base-distilled-patch16-224"
        model = AutoModel.from_pretrained(self.model_name).to(device)
        model.eval()

        self.intermediate_feature = {}

        if self.vit_layer >= 0:
            # 注册 hook 到 encoder 层
            try:
                encoder = model.base_model.encoder
            except AttributeError:
                encoder = model.vision_model.encoder  # fallback for some ViT-based models

            max_layer = len(encoder.layer)
            if self.vit_layer >= max_layer:
                raise ValueError(f"Invalid vit_layer={self.vit_layer}, must be in [0, {max_layer-1}]")
            
            encoder.layer[self.vit_layer].register_forward_hook(self._get_hook())

        return model

    def _get_hook(self):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.intermediate_feature['feature'] = output
        return hook_fn

    def extract_features(self, images, path=True, flatten=True):
        features = []

        for image in tqdm(images, desc="Extracting features with DeiT"):
            image = Image.open(image).convert("RGB") if path else image
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                if self.vit_layer >= 0:
                    _ = self.model(**inputs)
                    feat = self.intermediate_feature['feature']  # shape: (1, seq_len, dim)
                else:
                    outputs = self.model(**inputs)
                    if not hasattr(outputs, "last_hidden_state"):
                        raise ValueError("模型输出格式不正确")
                    feat = outputs.last_hidden_state  # 默认输出

                if flatten:
                    features.append(feat.flatten().cpu().numpy())
                else:
                    features.append(feat.squeeze(0).cpu().numpy())  # shape: (L, D)

        return np.stack(features, axis=0)
    
# class SAMFeatureExtractor(FeatureExtractor):

#     def __init__(self, **kwargs):
#         super().__init__('sam', **kwargs)
#         self.model = self.load_model()

#     def load_model(self):
#         """
#         Load the SAM model.
        
#         :return: SAM model instance.
#         """
#         # Placeholder for actual model loading logic
#         # This should involve loading the SAM model from a checkpoint or library
#         # ==== 加载 SAM 模型 ====
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         if not 'checkpoint_path' in self.kwargs:
#             raise ValueError("checkpoint_path is required for SAM model. You need to download the checkpoint from https://github.com/facebookresearch/segment-anything")
#         else:
#             sam_checkpoint = self.kwargs['checkpoint_path']
#         model_type = "vit_b"

        
#         # Load the SAM model
#         sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
#         return SamPredictor(sam)
    
#     def extract_features(self, images, path=True, flatten=True):
#         """
#         Extract features using the SAM model.
        
#         :param images: List of images to extract features from.
#         :return: Extracted features.
#         """
#         # Placeholder for actual feature extraction logic
#         # This should involve passing the images through the SAM model and obtaining the features
        
#         features = []
#         for image in tqdm(images, desc="Extracting features with SAM"):
#             # Apply the transformations to the image
#             image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) if path else ensure_cv2_format(image)
#             self.model.set_image(image)
#             # Pass the image through the model to get features
#             with torch.no_grad():
#                 feature = self.model.get_image_embedding().cpu()
#                 features.append(feature.squeeze())
        
#         if flatten:
#             features = np.array([feature.flatten() for feature in features])
#         else:
#             features = np.array([feature.permute(1,2,0).reshape(-1,256) for feature in features])

#         return features  # (N, 64*64, 256) or (N, 256*64*64)

class SAMFeatureExtractor(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__('sam', **kwargs)
        self.vit_layer = kwargs.get("vit_layer", -1)  # -1 表示默认最后一层
        self.model = self.load_model()

    def load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if 'checkpoint_path' not in self.kwargs:
            raise ValueError("checkpoint_path is required for SAM model.")
        sam_checkpoint = self.kwargs['checkpoint_path']
        model_type = "vit_b"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        self.intermediate_feature = {}

        # 注册 hook 到指定 ViT 层
        if self.vit_layer >= 0:
            if self.vit_layer >= len(sam.image_encoder.blocks):
                raise ValueError(f"Invalid layer index {self.vit_layer}, should be in [0, {len(sam.image_encoder.blocks)-1}]")
            sam.image_encoder.blocks[self.vit_layer].register_forward_hook(self._get_hook())
        else:
            # 使用默认输出
            self.intermediate_feature['feature'] = None

        predictor = SamPredictor(sam)
        return predictor

    def _get_hook(self):
        def hook_fn(module, input, output):
            self.intermediate_feature['feature'] = output
        return hook_fn

    def extract_features(self, images, path=True, flatten=True):
        features = []
        for image in tqdm(images, desc="Extracting features with SAM"):
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) if path else ensure_cv2_format(image)
            self.model.set_image(img)

            with torch.no_grad():
                # 触发 image_encoder.forward, 提取中间层特征
                if self.vit_layer >= 0:
                    # self.model.set_image(img)   
                    _ = self.model.get_image_embedding()  # 正确 resize + padding + forward
                    feat = self.intermediate_feature['feature'].cpu()  # shape: (1, L, C)
                    feat = feat.squeeze(0)  # shape: (L, C)
                else:
                    # 使用默认最终输出（get_image_embedding）
                    feat = self.model.get_image_embedding().cpu().squeeze(0).permute(1, 2, 0).reshape(-1, 256)

                features.append(feat)

        if flatten:
            features = np.array([f.flatten() for f in features])
        else:
            features = np.array([f for f in features])  # shape: (N, L, C)

        return features
    

class FeatureExtractorFactory:

    """
    Factory class to create feature extractor instances based on the provided model name.
    """
    @staticmethod
    def create_feature_extractor(model_name: str, **kwargs) -> FeatureExtractor:
        """
        Create a feature extractor instance based on the provided model name.
        
        :param model_name: Name of the model to use for feature extraction.
        :param kwargs: Additional arguments for the feature extractor.
        :return: Feature extractor instance.
        """
        if model_name == 'identity':
            return IdentityFeatureExtractor(**kwargs)
        elif model_name.startswith('dinov2'):
            if '-' in model_name:
                tokens = model_name.split('-')
                if len(tokens) == 3 and tokens[1] == 'vit_layer':
                    layer_id = int(tokens[2])
                    kwargs['layer_id'] = layer_id
            return DINOv2FeatureExtractor(**kwargs)
        elif model_name == 'stable_diffusion':
            return StableDiffusionFeatureExtractor(**kwargs)
        elif model_name.startswith('clip'):
            if '-' in model_name:
                tokens = model_name.split('-')
                if len(tokens) == 3 and tokens[1] == 'vit_layer':
                    layer_id = int(tokens[2])
                    kwargs['vit_layer'] = layer_id
            return CLIPFeatureExtractor(**kwargs)
        elif model_name.startswith('deit'):
            if '-' in model_name:
                tokens = model_name.split('-')
                if len(tokens) == 3 and tokens[1] == 'vit_layer':
                    layer_id = int(tokens[2])
                    kwargs['vit_layer'] = layer_id
            return DeitFeatureExtractor(**kwargs)
        elif model_name.startswith('sam'):
            if '-' in model_name:
                tokens = model_name.split('-')
                if len(tokens) == 3 and tokens[1] == 'vit_layer':
                    layer_id = int(tokens[2])
                    kwargs['vit_layer'] = layer_id
            return SAMFeatureExtractor(**kwargs)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")


