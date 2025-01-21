import torch
import numpy as np
from PIL import Image
import os
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from .utils import pad_and_resize, unpad_and_resize, resize_image, tensor_to_pil, pil_to_tensor, batched_pil_to_tensor
from .src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from .src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from .src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
from .src.pose_guider import PoseGuider
from .preprocess.humanparsing.run_parsing import Parsing
from .preprocess.dwpose import DWposeDetector
from .src.utils_mask import get_mask_location
import folder_paths

class FitDiTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": ("STRING", {"default": "cuda"}),
                "with_fp16": ("BOOLEAN", {"default": False}),
                "with_offload": ("BOOLEAN", {"default": False}),
                "with_aggressive_offload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("FITDIT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "FitDiT"

    def load_model(self, device="cuda", with_fp16=False, with_offload=False, with_aggressive_offload=False):
        weight_dtype = torch.float16 if with_fp16 else torch.bfloat16
        
        # Load model components
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
            os.path.join(folder_paths.models_dir, "FitDiT_models", "transformer_garm"), 
            torch_dtype=weight_dtype
        )
        
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
            os.path.join(folder_paths.models_dir, "FitDiT_models", "transformer_vton"), 
            torch_dtype=weight_dtype
        )
        
        pose_guider = PoseGuider(
            conditioning_embedding_channels=1536,
            conditioning_channels=3,
            block_out_channels=(32, 64, 256, 512)
        )
        pose_guider.load_state_dict(
            torch.load(os.path.join(folder_paths.models_dir, "FitDiT_models", "pose_guider", "diffusion_pytorch_model.bin"))
        )
        
        # Load CLIP models
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(folder_paths.models_dir, "clip", "clip-vit-large-patch14"), 
            torch_dtype=weight_dtype
        )
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(folder_paths.models_dir, "clip", "CLIP-ViT-bigG-14-laion2B-39B-b160k"), 
            torch_dtype=weight_dtype
        )
        
        # Move models to specified device
        pose_guider.to(device=device, dtype=weight_dtype)
        image_encoder_large.to(device=device)
        image_encoder_bigG.to(device=device)
        
        # Create pipeline
        pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            os.path.join(folder_paths.models_dir, "FitDiT_models"),
            torch_dtype=weight_dtype,
            transformer_garm=transformer_garm,
            transformer_vton=transformer_vton,
            pose_guider=pose_guider,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG
        )
        pipeline.to(device)
        if with_aggressive_offload:
            pipeline.enable_sequential_cpu_offload()
        elif with_offload:
            pipeline.enable_model_cpu_offload()
        pipeline.dwprocessor = DWposeDetector(model_root=os.path.join(folder_paths.models_dir, "FitDiT_models"), device='cpu')
        pipeline.parsing_model = Parsing(model_root=os.path.join(folder_paths.models_dir, "FitDiT_models"), device='cpu')
        return (pipeline,)

class FitDiTMaskGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FITDIT_MODEL",),
                "vton_image": ("IMAGE",),
                "category": (["Upper-body", "Lower-body", "Dresses"],),
                "offset_top": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_bottom": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_left": ("INT", {"default": 0, "min": -200, "max": 200}),
                "offset_right": ("INT", {"default": 0, "min": -200, "max": 200}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("masked_image", "mask", "pose_image")
    FUNCTION = "generate_mask"
    CATEGORY = "FitDiT"

    def generate_mask(self, model, vton_image, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            # Convert input image format
            vton_img = tensor_to_pil(vton_image)
            vton_img_det = resize_image(vton_img)
            
            # Generate pose information
            pose_image, keypoints, _, candidate = model.dwprocessor(np.array(vton_img_det)[:,:,::-1])
            candidate[candidate<0]=0
            candidate = candidate[0]
            
            candidate[:, 0]*=vton_img_det.width
            candidate[:, 1]*=vton_img_det.height
            
            # Process pose image
            pose_image = pose_image[:,:,::-1]
            pose_image = Image.fromarray(pose_image)
            
            # Generate parsing results
            model_parse, _ = model.parsing_model(vton_img_det)
            
            # Generate mask
            mask, mask_gray = get_mask_location(
                category, 
                model_parse,
                candidate, 
                model_parse.width, 
                model_parse.height,
                offset_top, 
                offset_bottom, 
                offset_left, 
                offset_right
            )
            
            # Resize masks
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            # Composite masked image
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)
            return (pil_to_tensor(masked_vton_img), pil_to_tensor(mask.convert("RGB")), pil_to_tensor(pose_image))

class FitDiTTryOn:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FITDIT_MODEL",),
                "vton_image": ("IMAGE",),
                "garm_image": ("IMAGE",),
                "mask": ("IMAGE",),
                "pose_image": ("IMAGE",),
                "n_steps": ("INT", {"default": 20, "min": 15, "max": 30}),
                "image_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0}),
                "seed": ("INT", {"default": -1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "resolution": (["768x1024", "1152x1536", "1536x2048"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "process"
    CATEGORY = "FitDiT"

    def process(self, model, vton_image, garm_image, mask, pose_image, n_steps, image_scale, seed, num_images, resolution):
        with torch.inference_mode():
            # Convert input image formats
            garm_img = tensor_to_pil(garm_image)
            vton_img = tensor_to_pil(vton_image)
            
            # Parse resolution
            new_width, new_height = resolution.split("x")
            new_width = int(new_width)
            new_height = int(new_height)
            
            # Resize images
            model_image_size = vton_img.size
            garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
            vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)
            
            # Process mask
            mask = tensor_to_pil(mask)
            mask, _, _ = pad_and_resize(mask, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
            mask = mask.convert("L")
            
            # Process pose image
            pose_image = tensor_to_pil(pose_image)
            pose_image, _, _ = pad_and_resize(pose_image, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
            
            # Set random seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            # Generate results
            results = model(
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask,
                pose_image=pose_image,
                num_images_per_prompt=num_images
            ).images
            
            # Process result images
            for idx in range(len(results)):
                results[idx] = unpad_and_resize(results[idx], pad_w, pad_h, model_image_size[0], model_image_size[1])
            return (batched_pil_to_tensor(results),) 