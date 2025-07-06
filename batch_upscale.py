#!/usr/bin/env python3
"""
Production-ready batch upscale script with transparency preservation
Splits RGBA into RGB+Alpha, upscales separately, then recombines
"""
import os
import glob
import requests
import time
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

# Handle PIL version compatibility
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

API_BASE_URL = "http://localhost:8888"
INPUT_DIR = "examples"
OUTPUT_DIR = "output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_minimal_prompt_for_texture(filename):
    """Generate minimal prompts focused on PRESERVATION, not transformation."""
    filename_lower = filename.lower()
    
    # VERY minimal prompts to avoid over-interpretation
    if 'dust' in filename_lower:
        return "dust texture"
    elif 'heart' in filename_lower:
        return "heart icon"
    elif 'magic' in filename_lower:
        return "potion bottle"
    elif 'arrow' in filename_lower:
        return "arrow"
    elif 'bomb' in filename_lower:
        return "bomb"
    elif 'deku' in filename_lower:
        return "game item"
    elif 'key' in filename_lower:
        return "key"
    elif 'door' in filename_lower or 'metal' in filename_lower:
        return "metal texture"
    elif 'explosion' in filename_lower or 'flame' in filename_lower:
        return "flame effect"
    else:
        return "game texture"

def check_transparency(input_path):
    """Check if image has transparency (alpha channel)."""
    try:
        with Image.open(input_path) as img:
            return img.mode in ('RGBA', 'LA') or 'transparency' in img.info
    except:
        return False

def split_rgba_image(image_path):
    """Split RGBA image into RGB and Alpha components."""
    with Image.open(image_path) as img:
        if img.mode != 'RGBA':
            # Convert to RGBA if it has transparency info
            if 'transparency' in img.info:
                img = img.convert('RGBA')
            else:
                # No transparency, return RGB only
                rgb_img = img.convert('RGB')
                return rgb_img, None
        
        # Split RGBA into RGB and Alpha
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))  # White background
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha as mask
        
        alpha_img = img.split()[3]  # Extract alpha channel
        
        return rgb_img, alpha_img

def upscale_rgb_with_api(rgb_image, prompt):
    """Upscale RGB image using the API."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        rgb_image.save(temp_file.name, 'PNG')
        temp_path = temp_file.name
    
    try:
        with open(temp_path, 'rb') as f:
            files = {'file': f}
            # Research-optimized parameters
            data = {
                'prompt': prompt,
                'noise_level': 0.25,  # Higher noise preserves original structure
                'num_inference_steps': 25,  # Optimal for SDXL upscaling
                'guidance_scale': 7.5  # Sweet spot for upscaling
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/upscale",
                files=files,
                data=data,
                timeout=180  # 3 minute timeout
            )
            
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as result_file:
                    result_file.write(response.content)
                    result_path = result_file.name
                
                upscaled_rgb = Image.open(result_path)
                os.unlink(result_path)  # Clean up temp file
                os.unlink(temp_path)   # Clean up temp file
                
                return True, upscaled_rgb
            else:
                os.unlink(temp_path)
                return False, f"HTTP {response.status_code}: {response.text}"
                
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return False, str(e)

def upscale_alpha_traditional(alpha_image, scale_factor=4):
    """Upscale alpha channel using traditional high-quality algorithm."""
    # Use Lanczos for high-quality alpha upscaling
    new_size = (alpha_image.width * scale_factor, alpha_image.height * scale_factor)
    upscaled_alpha = alpha_image.resize(new_size, LANCZOS)
    return upscaled_alpha

def combine_rgb_alpha(rgb_image, alpha_image):
    """Combine upscaled RGB and Alpha back into RGBA."""
    # Ensure both images are the same size
    if rgb_image.size != alpha_image.size:
        # Resize alpha to match RGB (in case of slight size differences)
        alpha_image = alpha_image.resize(rgb_image.size, LANCZOS)
    
    # Convert RGB to RGBA and apply alpha
    rgba_image = rgb_image.convert('RGBA')
    rgba_data = list(rgba_image.getdata())
    alpha_data = list(alpha_image.getdata())
    
    # Combine RGB with Alpha
    combined_data = []
    for i in range(len(rgba_data)):
        r, g, b, _ = rgba_data[i]
        a = alpha_data[i] if isinstance(alpha_data[i], int) else alpha_data[i][0]
        combined_data.append((r, g, b, a))
    
    result_image = Image.new('RGBA', rgb_image.size)
    result_image.putdata(combined_data)
    
    return result_image

def upscale_image_with_transparency(input_path, output_path, prompt):
    """Upscale image preserving transparency by processing RGB and Alpha separately."""
    try:
        # Check if image has transparency
        has_transparency = check_transparency(input_path)
        
        if not has_transparency:
            # No transparency, process normally
            with open(input_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'prompt': prompt,
                    'noise_level': 0.25,
                    'num_inference_steps': 25,
                    'guidance_scale': 7.5
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/upscale",
                    files=files,
                    data=data,
                    timeout=180
                )
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as out_f:
                        out_f.write(response.content)
                    return True, len(response.content), "standard"
                else:
                    return False, f"HTTP {response.status_code}: {response.text}", "error"
        else:
            # Has transparency, split and process separately
            print(f"      🔧 Splitting RGBA into RGB + Alpha...")
            rgb_img, alpha_img = split_rgba_image(input_path)
            
            if alpha_img is None:
                # Fallback to standard processing
                return upscale_image_with_transparency(input_path, output_path, prompt)
            
            print(f"      🎨 Upscaling RGB component with API...")
            success, result = upscale_rgb_with_api(rgb_img, prompt)
            if not success:
                return False, result, "rgb_failed"
            
            upscaled_rgb = result
            
            print(f"      🖼️  Upscaling Alpha component with Lanczos...")
            upscaled_alpha = upscale_alpha_traditional(alpha_img, scale_factor=4)
            
            print(f"      🔗 Combining RGB + Alpha into final RGBA...")
            final_rgba = combine_rgb_alpha(upscaled_rgb, upscaled_alpha)
            
            # Save final result
            final_rgba.save(output_path, 'PNG')
            
            # Calculate file size
            file_size = os.path.getsize(output_path)
            
            return True, file_size, "transparency_preserved"
                
    except Exception as e:
        return False, str(e), "error"

def main():
    """Main batch processing function with transparency preservation."""
    # Get all original PNG files (skip processed ones)
    png_files = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    png_files = [f for f in png_files if not any(suffix in f for suffix in ["_4x", "_hq", "_std", "_cons", "_opt", "_trans"])]
    
    if not png_files:
        print(f"No unprocessed PNG files found in {INPUT_DIR} directory!")
        return
    
    # Check for transparency
    transparent_files = []
    standard_files = []
    for file in png_files:
        if check_transparency(file):
            transparent_files.append(os.path.basename(file))
        else:
            standard_files.append(os.path.basename(file))
    
    print(f"🎮 TRANSPARENCY-PRESERVING BATCH PROCESSING")
    print(f"📊 Found {len(png_files)} texture files to process")
    print(f"   • {len(transparent_files)} files with transparency (will preserve)")
    print(f"   • {len(standard_files)} files without transparency (standard processing)")
    print(f"📁 Input: {INPUT_DIR}/")
    print(f"📁 Output: {OUTPUT_DIR}/")
    print(f"🌐 API: {API_BASE_URL}")
    print("⚡ OPTIMIZED SETTINGS: 25 steps, 7.5 guidance, 0.25 noise")
    print("🔧 TRANSPARENCY METHOD: Split RGBA → Upscale RGB+Alpha → Recombine")
    print("=" * 70)
    
    # Check API health
    try:
        health_response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if not health_data.get('model_loaded', False):
                print("⚠️  Warning: Model not loaded yet, this may take longer...")
        else:
            print("❌ API health check failed!")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    total_files = len(png_files)
    successful = 0
    failed = 0
    transparency_preserved = 0
    total_start_time = time.time()
    
    for i, input_path in enumerate(png_files, 1):
        filename = os.path.basename(input_path)
        # Create output filename by inserting "_4x" before the extension
        name_part, ext = os.path.splitext(filename)
        output_filename = f"{name_part}_4x{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"⏭️  [{i:2d}/{total_files}] Skipping {filename} (already exists)")
            successful += 1
            continue
        
        prompt = get_minimal_prompt_for_texture(filename)
        has_transparency = check_transparency(input_path)
        
        print(f"🔄 [{i:2d}/{total_files}] Processing {filename} → {output_filename}")
        if has_transparency:
            print(f"   ✨ Transparency detected - using RGB+Alpha method")
        print(f"   💭 Minimal Prompt: {prompt}")
        
        start_time = time.time()
        success, result, method = upscale_image_with_transparency(input_path, output_path, prompt)
        end_time = time.time()
        
        if success:
            file_size_kb = int(result) / 1024
            processing_time = end_time - start_time
            if method == "transparency_preserved":
                print(f"   ✅ Success! {file_size_kb:.1f}KB in {processing_time:.1f}s (transparency preserved)")
                transparency_preserved += 1
            else:
                print(f"   ✅ Success! {file_size_kb:.1f}KB in {processing_time:.1f}s (standard)")
            successful += 1
        else:
            print(f"   ❌ Failed ({method}): {result}")
            failed += 1
        
        print()  # Empty line for readability
    
    total_time = time.time() - total_start_time
    
    print("=" * 70)
    print(f"🎯 TRANSPARENCY-PRESERVING PROCESSING COMPLETE!")
    print(f"✅ Successful: {successful}/{total_files}")
    print(f"❌ Failed: {failed}/{total_files}")
    print(f"✨ Transparency preserved: {transparency_preserved} files")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    print(f"📁 Enhanced files saved in: {OUTPUT_DIR}/")
    print(f"📝 Naming convention: original_name_4x.png")
    print(f"🔧 Method: RGB+Alpha split for transparent images")
    print(f"⚡ Settings: 25 steps, 7.5 guidance, 0.25 noise (research-optimized)")
    
    if successful > 0:
        print(f"\n📊 Average processing time: {total_time/total_files:.1f}s per image")
        if transparency_preserved > 0:
            print(f"🌟 Successfully preserved transparency in {transparency_preserved} images!")

if __name__ == "__main__":
    main() 