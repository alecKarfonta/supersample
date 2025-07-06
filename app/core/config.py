import os
from typing import Optional
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings with environment variable support."""
    
    # API Configuration
    api_title: str = "Image Super Sampling API"
    api_version: str = "1.0.0"
    api_description: str = "4x image upscaling API using Stable Diffusion"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Model Configuration
    model_id: str = "stabilityai/stable-diffusion-x4-upscaler"
    device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    torch_dtype: str = "float16"  # or "float32"
    enable_attention_slicing: bool = True
    enable_cpu_offload: bool = False
    
    # File Processing
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: list = ["PNG", "JPEG", "JPG", "WEBP"]
    max_input_size: int = 512  # Max dimension for input images
    
    # API Limits
    max_concurrent_requests: int = 3
    request_timeout: int = 60
    rate_limit_per_minute: int = 10
    
    # Inference Parameters
    default_noise_level: float = 0.2
    min_noise_level: float = 0.0
    max_noise_level: float = 1.0
    default_num_inference_steps: int = 20
    default_guidance_scale: float = 7.5
    
    # Storage
    temp_dir: str = "/tmp/supersample"
    cache_dir: str = "./cache"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_prefix = "SUPERSAMPLE_"


# Global settings instance
settings = Settings() 