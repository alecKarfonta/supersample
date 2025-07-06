#!/usr/bin/env python3
"""
Simple script to run the Image Super Sampling API
"""
import os
import sys
import subprocess
import argparse


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import diffusers
        import fastapi
        import uvicorn
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Device count: {gpu_count})")
            return True
        else:
            print("⚠ GPU not available, will use CPU (slower)")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def main():
    """Main function to run the API."""
    parser = argparse.ArgumentParser(description="Run Image Super Sampling API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Image Super Sampling API")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        check_gpu()
    
    # Set environment variables
    os.environ.setdefault("SUPERSAMPLE_HOST", args.host)
    os.environ.setdefault("SUPERSAMPLE_PORT", str(args.port))
    
    # Run the API
    print(f"\nStarting API on {args.host}:{args.port}")
    print(f"Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/api/v1/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", args.host,
            "--port", str(args.port),
            "--log-level", args.log_level
        ]
        
        if args.reload:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 