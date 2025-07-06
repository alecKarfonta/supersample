# Image Super Sampling API

A production-ready API for image super sampling using Stable Diffusion x4 upscaler with transparency preservation.

## 🚀 Features

- **AI-Powered Upscaling**: Uses Stable Diffusion x4 upscaler for high-quality image enhancement
- **Transparency Preservation**: Smart RGB+Alpha splitting to maintain transparency in game textures
- **GPU Acceleration**: Full CUDA support for fast processing
- **RESTful API**: Clean FastAPI endpoints for easy integration
- **Batch Processing**: Command-line tools for processing multiple images
- **Research-Optimized**: Parameters tuned based on SDXL research for optimal results

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Docker & Docker Compose
- 8GB+ VRAM for optimal performance

## 🛠️ Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/alecKarfonta/supersample.git
cd supersample

# Build and start the API
docker compose up -d --build

# The API will be available at http://localhost:8888
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
python run.py
```

## 🎮 Usage

### API Endpoints

#### Upscale Image
```bash
curl -X POST "http://localhost:8888/api/v1/upscale" \
  -F "file=@your_image.png" \
  -F "prompt=game texture" \
  -F "noise_level=0.25" \
  -F "num_inference_steps=25" \
  -F "guidance_scale=7.5"
```

#### Health Check
```bash
curl "http://localhost:8888/api/v1/health"
```

#### Model Information
```bash
curl "http://localhost:8888/api/v1/model-info"
```

### Batch Processing

```bash
# Process all images in examples/ folder
python batch_upscale.py

# Output will be saved to output/ folder
```

## 🔧 Configuration

### Environment Variables

- `MODEL_ID`: HuggingFace model ID (default: "stabilityai/stable-diffusion-x4-upscaler")
- `DEVICE`: Device to use ("cuda" or "cpu")
- `MAX_IMAGE_SIZE`: Maximum input image size in pixels
- `API_HOST`: API host (default: "0.0.0.0")
- `API_PORT`: API port (default: 8888)

### Docker Configuration

The `docker-compose.yml` file includes:
- GPU support with NVIDIA runtime
- Volume mounts for cache and temp directories
- Health checks
- Port mapping (8888)

## 🎯 Transparency Preservation

The API includes a sophisticated transparency preservation system:

1. **Detection**: Automatically detects images with alpha channels
2. **Splitting**: Separates RGB and Alpha components
3. **Processing**: 
   - RGB: Enhanced with Stable Diffusion
   - Alpha: Upscaled with high-quality Lanczos interpolation
4. **Recombination**: Combines enhanced RGB with preserved Alpha

This ensures game textures maintain their transparency behavior while getting AI enhancement.

## 📊 Performance

- **Processing Speed**: ~1.2s per image (GPU)
- **Memory Usage**: ~1.6GB VRAM during operation
- **Model Size**: ~5.7GB (downloaded once)
- **Upscale Factor**: 4x (32x32 → 128x128)

## 🧪 Testing

```bash
# Run API tests
python -m pytest app/tests/

# Manual testing
python test_client.py
```

## 📁 Project Structure

```
supersample/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   │   └── config.py          # Configuration management
│   │   └── schemas.py         # Pydantic models
│   │   └── model_manager.py   # Model loading & inference
│   └── main.py                # FastAPI application
├── examples/                  # Input images
├── output/                    # Processed images
├── batch_upscale.py          # Batch processing script
├── docker-compose.yml        # Docker configuration
├── Dockerfile                # Container definition
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔬 Research & Optimization

The API uses research-optimized parameters based on SDXL studies:

- **Steps**: 25 (optimal for upscaling)
- **Guidance Scale**: 7.5 (sweet spot for preservation)
- **Noise Level**: 0.25 (preserves original structure)
- **Scheduler**: UniPC (fast and stable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) for the Stable Diffusion x4 upscaler
- [Hugging Face](https://huggingface.co/) for model hosting
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the API health endpoint for system status
- Review the logs: `docker compose logs -f` 