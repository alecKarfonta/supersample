import os
import logging
import torch
import gc
import time
from typing import Optional, Dict, Any
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the Stable Diffusion upscaling model pipeline."""
    
    def __init__(self):
        self.pipeline: Optional[StableDiffusionUpscalePipeline] = None
        self.model_loaded = False
        self.device = settings.device
        self.model_id = settings.model_id
        
    def load_model(self) -> bool:
        """Load the model pipeline."""
        try:
            logger.info(f"Loading model {self.model_id} on device {self.device}")
            
            # Set torch dtype based on configuration
            torch_dtype = torch.float16 if settings.torch_dtype == "float16" else torch.float32
            
            # Load the pipeline
            self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                logger.info(f"Model loaded on CUDA device")
            else:
                self.pipeline = self.pipeline.to("cpu")
                logger.info(f"Model loaded on CPU")
                
            # Enable optimizations
            if settings.enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled")
                
            if settings.enable_cpu_offload and self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                logger.info("CPU offload enabled")
                
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
            # Warm up the model
            self._warmup_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model_loaded = False
            return False
            
    def _warmup_model(self):
        """Warm up the model with a dummy inference."""
        try:
            logger.info("Warming up model...")
            # Create a small dummy image
            dummy_image = Image.new('RGB', (64, 64), (255, 255, 255))
            dummy_prompt = "test"
            
            # Run inference with minimal steps
            with torch.no_grad():
                _ = self.pipeline(
                    prompt=dummy_prompt,
                    image=dummy_image,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    noise_level=0.0
                )
                
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
            
    def upscale_image(
        self,
        image: Image.Image,
        prompt: str,
        noise_level: float = 0.2,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> tuple[Image.Image, Dict[str, Any]]:
        """
        Upscale an image using the loaded model.
        
        Args:
            image: Input PIL image
            prompt: Text prompt for guidance
            noise_level: Noise level (0.0-1.0)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            
        Returns:
            Tuple of (upscaled_image, metadata)
        """
        if not self.model_loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded")
            
        try:
            start_time = time.time()
            
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            original_size = image.size
            
            # Validate image size
            max_dim = max(original_size)
            if max_dim > settings.max_input_size:
                # Resize while maintaining aspect ratio
                scale = settings.max_input_size / max_dim
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {original_size} to {image.size}")
                
            logger.info(f"Starting upscaling with prompt: '{prompt}'")
            
            # Run inference
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    noise_level=noise_level
                )
                
            upscaled_image = result.images[0]
            processing_time = time.time() - start_time
            
            # Cleanup GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                
            metadata = {
                "original_size": original_size,
                "upscaled_size": upscaled_image.size,
                "processing_time": processing_time,
                "prompt": prompt,
                "noise_level": noise_level,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            
            logger.info(f"Upscaling completed in {processing_time:.2f}s")
            
            return upscaled_image, metadata
            
        except Exception as e:
            logger.error(f"Upscaling failed: {str(e)}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        gpu_available = torch.cuda.is_available()
        memory_info = {}
        
        if gpu_available and self.device == "cuda":
            try:
                memory_info = {
                    "gpu_memory_used": f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
                    "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
                }
            except:
                pass
                
        return {
            "model_id": self.model_id,
            "model_type": "StableDiffusionUpscalePipeline",
            "device": self.device,
            "model_loaded": self.model_loaded,
            "gpu_available": gpu_available,
            "capabilities": ["4x upscaling", "text-guided enhancement"],
            "max_input_size": settings.max_input_size,
            "upscale_factor": 4,
            "memory_usage": memory_info
        }
        
    def cleanup(self):
        """Cleanup model resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        gc.collect()
        self.model_loaded = False
        logger.info("Model cleanup completed")


# Global model manager instance
model_manager = ModelManager() 