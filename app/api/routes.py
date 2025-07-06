import io
import logging
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from PIL import Image
from app.models.schemas import UpscaleResponse, ModelInfo, HealthResponse, ErrorResponse
from app.services.model_manager import model_manager
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


async def validate_image_file(file: UploadFile) -> Image.Image:
    """Validate and load uploaded image file."""
    # Check file size
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size / 1024 / 1024:.1f}MB"
        )
    
    # Read file content
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image format
        if image.format not in settings.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format. Supported formats: {', '.join(settings.supported_formats)}"
            )
            
        return image
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )


@router.post(
    "/upscale",
    response_model=UpscaleResponse,
    summary="Upscale an image",
    description="Upload an image and get a 4x upscaled version using text guidance"
)
async def upscale_image(
    file: UploadFile = File(..., description="Image file to upscale"),
    prompt: str = Form(..., description="Text prompt to guide the upscaling"),
    noise_level: Optional[float] = Form(
        default=settings.default_noise_level,
        description="Noise level (0.0-1.0)"
    ),
    num_inference_steps: Optional[int] = Form(
        default=settings.default_num_inference_steps,
        description="Number of inference steps"
    ),
    guidance_scale: Optional[float] = Form(
        default=settings.default_guidance_scale,
        description="Guidance scale"
    )
):
    """Upscale an image using the Stable Diffusion x4 upscaler."""
    
    # Validate parameters and set defaults
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Set defaults if None
    if noise_level is None:
        noise_level = settings.default_noise_level
    if num_inference_steps is None:
        num_inference_steps = settings.default_num_inference_steps
    if guidance_scale is None:
        guidance_scale = settings.default_guidance_scale
    
    if not (settings.min_noise_level <= noise_level <= settings.max_noise_level):
        raise HTTPException(
            status_code=400,
            detail=f"Noise level must be between {settings.min_noise_level} and {settings.max_noise_level}"
        )
    
    if not (1 <= num_inference_steps <= 100):
        raise HTTPException(
            status_code=400,
            detail="Number of inference steps must be between 1 and 100"
        )
    
    if not (1.0 <= guidance_scale <= 20.0):
        raise HTTPException(
            status_code=400,
            detail="Guidance scale must be between 1.0 and 20.0"
        )
    
    # Check if model is loaded
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the service to initialize."
        )
    
    try:
        # Validate and load image
        image = await validate_image_file(file)
        
        # Run upscaling in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        upscaled_image, metadata = await loop.run_in_executor(
            None,
            model_manager.upscale_image,
            image,
            prompt,
            noise_level,
            num_inference_steps,
            guidance_scale
        )
        
        # Convert upscaled image to bytes for response
        img_buffer = io.BytesIO()
        upscaled_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Return image as streaming response
        return StreamingResponse(
            io.BytesIO(img_buffer.getvalue()),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=upscaled_{file.filename}",
                "X-Original-Size": f"{metadata['original_size'][0]}x{metadata['original_size'][1]}",
                "X-Upscaled-Size": f"{metadata['upscaled_size'][0]}x{metadata['upscaled_size'][1]}",
                "X-Processing-Time": f"{metadata['processing_time']:.2f}",
                "X-Prompt": prompt,
                "X-Noise-Level": str(noise_level),
                "X-Inference-Steps": str(num_inference_steps),
                "X-Guidance-Scale": str(guidance_scale)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upscaling failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upscaling failed: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API service"
)
async def health_check():
    """Health check endpoint."""
    model_info = model_manager.get_model_info()
    
    return HealthResponse(
        status="healthy" if model_manager.model_loaded else "initializing",
        model_loaded=model_info["model_loaded"],
        gpu_available=model_info["gpu_available"],
        memory_usage=model_info["memory_usage"]
    )


@router.get(
    "/model-info",
    response_model=ModelInfo,
    summary="Model information",
    description="Get information about the loaded model"
)
async def get_model_info():
    """Get model information."""
    model_info = model_manager.get_model_info()
    
    return ModelInfo(
        model_id=model_info["model_id"],
        model_type=model_info["model_type"],
        device=model_info["device"],
        capabilities=model_info["capabilities"],
        max_input_size=model_info["max_input_size"],
        upscale_factor=model_info["upscale_factor"]
    ) 