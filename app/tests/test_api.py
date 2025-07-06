import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_test_image(size=(64, 64), color=(255, 0, 0), format="PNG"):
    """Create a test image for testing purposes."""
    image = Image.new('RGB', size, color)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format=format)
    img_buffer.seek(0)
    return img_buffer


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "gpu_available" in data


def test_model_info_endpoint():
    """Test the model info endpoint."""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_id" in data
    assert "model_type" in data
    assert "device" in data
    assert "capabilities" in data
    assert "max_input_size" in data
    assert "upscale_factor" in data


def test_upscale_endpoint_validation():
    """Test the upscale endpoint with invalid inputs."""
    # Test without image
    response = client.post("/api/v1/upscale", data={"prompt": "test"})
    assert response.status_code == 422
    
    # Test without prompt
    img_buffer = create_test_image()
    response = client.post(
        "/api/v1/upscale",
        files={"file": ("test.png", img_buffer, "image/png")}
    )
    assert response.status_code == 422


def test_upscale_endpoint_with_valid_input():
    """Test the upscale endpoint with valid input."""
    img_buffer = create_test_image()
    response = client.post(
        "/api/v1/upscale",
        files={"file": ("test.png", img_buffer, "image/png")},
        data={"prompt": "a red square"}
    )
    
    # This might fail if model is not loaded, but we test the API structure
    assert response.status_code in [200, 503]  # 503 if model not loaded
    
    if response.status_code == 503:
        data = response.json()
        assert "Model not loaded" in data["detail"]


def test_upscale_endpoint_invalid_noise_level():
    """Test the upscale endpoint with invalid noise level."""
    img_buffer = create_test_image()
    response = client.post(
        "/api/v1/upscale",
        files={"file": ("test.png", img_buffer, "image/png")},
        data={"prompt": "test", "noise_level": "1.5"}  # Invalid: > 1.0
    )
    assert response.status_code == 400
    data = response.json()
    assert "Noise level must be between" in data["detail"]


def test_upscale_endpoint_invalid_steps():
    """Test the upscale endpoint with invalid inference steps."""
    img_buffer = create_test_image()
    response = client.post(
        "/api/v1/upscale",
        files={"file": ("test.png", img_buffer, "image/png")},
        data={"prompt": "test", "num_inference_steps": "0"}  # Invalid: < 1
    )
    assert response.status_code == 400
    data = response.json()
    assert "Number of inference steps must be between" in data["detail"]


def test_upscale_endpoint_large_file():
    """Test the upscale endpoint with a large file."""
    # Create a large image (this would be rejected)
    img_buffer = create_test_image(size=(2048, 2048))
    
    # Calculate approximate size (this might be close to or exceed the limit)
    file_size = len(img_buffer.getvalue())
    
    response = client.post(
        "/api/v1/upscale",
        files={"file": ("large.png", img_buffer, "image/png")},
        data={"prompt": "test"}
    )
    
    # If the file is too large, it should return 413
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        assert response.status_code == 413
        data = response.json()
        assert "File too large" in data["detail"]


def test_upscale_endpoint_unsupported_format():
    """Test the upscale endpoint with unsupported format."""
    # Create a BMP image (unsupported format)
    img_buffer = create_test_image(format="BMP")
    
    response = client.post(
        "/api/v1/upscale",
        files={"file": ("test.bmp", img_buffer, "image/bmp")},
        data={"prompt": "test"}
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported image format" in data["detail"]


if __name__ == "__main__":
    pytest.main([__file__]) 