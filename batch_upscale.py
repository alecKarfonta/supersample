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
import argparse
import itertools
from pathlib import Path
from PIL import Image
import numpy as np

API_BASE_URL = "http://localhost:8888"
INPUT_DIR = "examples"
OUTPUT_DIR = "output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_list_argument(value):
    """Parse a comma-separated string into a list of values."""
    if not value:
        return []
    return [item.strip() for item in value.split(',')]

def parse_float_list(value):
    """Parse a comma-separated string into a list of floats."""
    if not value:
        return []
    return [float(item.strip()) for item in value.split(',')]

def parse_int_list(value):
    """Parse a comma-separated string into a list of integers."""
    if not value:
        return []
    return [int(item.strip()) for item in value.split(',')]

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
        rgb_img = Image.new('RGB', img.size)  # Create RGB image
        rgb_img.paste((255, 255, 255), (0, 0, img.size[0], img.size[1]))  # Fill with white
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha as mask
        
        alpha_img = img.split()[3]  # Extract alpha channel
        
        return rgb_img, alpha_img

def upscale_rgb_with_api(rgb_image, prompt, noise_level, num_inference_steps, guidance_scale):
    """Upscale RGB image using the API."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        rgb_image.save(temp_file.name, 'PNG')
        temp_path = temp_file.name
    
    try:
        with open(temp_path, 'rb') as f:
            files = {'file': f}
            data = {
                'prompt': prompt,
                'noise_level': noise_level,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale
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
    upscaled_alpha = alpha_image.resize(new_size, Image.Resampling.LANCZOS)
    return upscaled_alpha

def combine_rgb_alpha(rgb_image, alpha_image):
    """Combine upscaled RGB and Alpha back into RGBA."""
    # Ensure both images are the same size
    if rgb_image.size != alpha_image.size:
        # Resize alpha to match RGB (in case of slight size differences)
        alpha_image = alpha_image.resize(rgb_image.size, Image.Resampling.LANCZOS)
    
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

def upscale_image_with_transparency(input_path, output_path, prompt, noise_level, num_inference_steps, guidance_scale):
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
                    'noise_level': noise_level,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale
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
                return upscale_image_with_transparency(input_path, output_path, prompt, noise_level, num_inference_steps, guidance_scale)
            
            print(f"      🎨 Upscaling RGB component with API...")
            success, result = upscale_rgb_with_api(rgb_img, prompt, noise_level, num_inference_steps, guidance_scale)
            if not success:
                return False, result, "rgb_failed"
            
            upscaled_rgb = result
            
            print(f"      🖼️  Upscaling Alpha component with LANCZOS...")
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

def generate_parameter_combinations(args):
    """Generate all combinations of parameters for testing."""
    # Default values
    noise_levels = [0.25] if not args.noise_levels else args.noise_levels
    inference_steps = [25] if not args.inference_steps else args.inference_steps
    guidance_scales = [7.5] if not args.guidance_scales else args.guidance_scales
    prompts = ["game texture"] if not args.prompts else args.prompts
    
    # Generate all combinations
    combinations = list(itertools.product(noise_levels, inference_steps, guidance_scales, prompts))
    
    print(f"🧪 PARAMETER COMBINATIONS:")
    print(f"   • Noise Levels: {noise_levels}")
    print(f"   • Inference Steps: {inference_steps}")
    print(f"   • Guidance Scales: {guidance_scales}")
    print(f"   • Prompts: {prompts}")
    print(f"   • Total Combinations: {len(combinations)}")
    print()
    
    return combinations

def create_output_filename(base_name, noise_level, inference_steps, guidance_scale, prompt):
    """Create output filename with parameter information."""
    # Clean prompt for filename (remove special chars)
    clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_prompt = clean_prompt.replace(' ', '_')[:20]  # Limit length
    
    # Create parameter suffix
    param_suffix = f"_n{noise_level}_i{inference_steps}_g{guidance_scale}_{clean_prompt}"
    
    # Insert before extension
    name_part, ext = os.path.splitext(base_name)
    return f"{name_part}{param_suffix}{ext}"

def remove_white_halo(img: Image.Image) -> Image.Image:
    """
    Premultiplies RGB by alpha to remove white (or light) halos on transparent PNGs.
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    arr = np.array(img).astype(np.float32)
    alpha = arr[..., 3:4] / 255.0
    arr[..., :3] = arr[..., :3] * alpha
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')

def main():
    """Main batch processing function with transparency preservation and parameter testing."""
    # Declare globals at the start
    global INPUT_DIR, OUTPUT_DIR, API_BASE_URL
    
    parser = argparse.ArgumentParser(
        description="Batch upscale images with transparency preservation and parameter testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python batch_upscale.py

  # Test different noise levels
  python batch_upscale.py --noise-levels 0.1,0.25,0.5

  # Test multiple parameter combinations
  python batch_upscale.py --noise-levels 0.1,0.25 --inference-steps 20,25 --guidance-scales 7.0,7.5

  # Test different prompts
  python batch_upscale.py --prompts "game texture,detailed texture,high quality texture"

  # Full combinatorial testing
  python batch_upscale.py --noise-levels 0.1,0.25,0.5 --inference-steps 20,25,30 --guidance-scales 7.0,7.5,8.0 --prompts "game texture,detailed texture"
        """
    )
    
    # File and directory arguments
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Input directory (default: examples)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory (default: output)")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL (default: http://localhost:8888)")
    
    # Model generation parameters (comma-separated lists)
    parser.add_argument("--noise-levels", type=parse_float_list, help="Noise levels to test (comma-separated, e.g., 0.1,0.25,0.5)")
    parser.add_argument("--inference-steps", type=parse_int_list, help="Inference steps to test (comma-separated, e.g., 20,25,30)")
    parser.add_argument("--guidance-scales", type=parse_float_list, help="Guidance scales to test (comma-separated, e.g., 7.0,7.5,8.0)")
    parser.add_argument("--prompts", type=parse_list_argument, help="Prompts to test (comma-separated, e.g., 'game texture,detailed texture')")
    
    # Processing options
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of existing files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument('--remove-white-halo', action='store_true', help='Post-process output PNGs to remove white halos (premultiply RGB by alpha)')
    
    args = parser.parse_args()
    
    # Update global variables
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    API_BASE_URL = args.api_url
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all original PNG files (skip processed ones)
    png_files = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    png_files = [f for f in png_files if not any(suffix in f for suffix in ["_4x", "_hq", "_std", "_cons", "_opt", "_trans"])]
    
    if not png_files:
        print(f"No unprocessed PNG files found in {INPUT_DIR} directory!")
        return
    
    # Limit files if specified
    if args.max_files:
        png_files = png_files[:args.max_files]
    
    # Generate parameter combinations
    combinations = generate_parameter_combinations(args)
    
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
    total_combinations = len(combinations)
    total_operations = total_files * total_combinations
    
    successful = 0
    failed = 0
    transparency_preserved = 0
    total_start_time = time.time()
    
    operation_count = 0
    
    for i, input_path in enumerate(png_files, 1):
        filename = os.path.basename(input_path)
        has_transparency = check_transparency(input_path)
        
        print(f"🔄 [{i:2d}/{total_files}] Processing {filename}")
        if has_transparency:
            print(f"   ✨ Transparency detected - using RGB+Alpha method")
        
        for j, (noise_level, inference_steps, guidance_scale, prompt) in enumerate(combinations, 1):
            operation_count += 1
            
            # Create output filename with parameter information
            output_filename = create_output_filename(filename, noise_level, inference_steps, guidance_scale, prompt)
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Skip if already processed and not forcing
            if os.path.exists(output_path) and not args.force:
                print(f"   ⏭️  [{j:2d}/{total_combinations}] Skipping {output_filename} (already exists)")
                successful += 1
                continue
            
            print(f"   🔄 [{j:2d}/{total_combinations}] Testing: n={noise_level}, i={inference_steps}, g={guidance_scale}, p='{prompt}'")
            print(f"   💭 Output: {output_filename}")
            
            start_time = time.time()
            success, result, method = upscale_image_with_transparency(
                input_path, output_path, prompt, noise_level, inference_steps, guidance_scale
            )
            end_time = time.time()
            
            if success:
                if isinstance(result, (int, float)):
                    file_size_kb = result / 1024
                else:
                    file_size_kb = 0
                processing_time = end_time - start_time
                if method == "transparency_preserved":
                    print(f"      ✅ Success! {file_size_kb:.1f}KB in {processing_time:.1f}s (transparency preserved)")
                    transparency_preserved += 1
                else:
                    print(f"      ✅ Success! {file_size_kb:.1f}KB in {processing_time:.1f}s (standard)")
                successful += 1
                # After saving output_img, optionally remove white halo
                if args.remove_white_halo:
                    output_img = remove_white_halo(Image.open(output_path))
                    output_img.save(output_path)
            else:
                print(f"      ❌ Failed ({method}): {result}")
                failed += 1
        
        print()  # Empty line for readability
    
    total_time = time.time() - total_start_time
    
    print("=" * 70)
    print(f"🎯 PARAMETER TESTING COMPLETE!")
    print(f"✅ Successful: {successful}/{total_operations}")
    print(f"❌ Failed: {failed}/{total_operations}")
    print(f"✨ Transparency preserved: {transparency_preserved} operations")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    print(f"📁 Enhanced files saved in: {OUTPUT_DIR}/")
    print(f"📝 Naming convention: original_name_n{noise_level}_i{inference_steps}_g{guidance_scale}_{prompt}.png")
    print(f"🔧 Method: RGB+Alpha split for transparent images")
    
    if successful > 0:
        print(f"\n📊 Average processing time: {total_time/total_operations:.1f}s per operation")
        if transparency_preserved > 0:
            print(f"🌟 Successfully preserved transparency in {transparency_preserved} operations!")

if __name__ == "__main__":
    main() 