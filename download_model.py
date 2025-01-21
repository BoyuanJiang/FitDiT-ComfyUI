from huggingface_hub import snapshot_download
import os
from pathlib import Path
import argparse

def download_models(root_dir: str = "./"):
    """
    Download model weights to the specified directory
    
    Args:
        root_dir (str): Root directory for downloads, defaults to current directory
    """
    # Ensure root directory is an absolute path
    root_dir = os.path.abspath(root_dir)
    
    # Define model list with repository IDs, relative paths and ignore patterns
    models = [
        {
            "repo_id": "BoyuanJiang/FitDiT",
            "relative_path": "models/FitDiT_models",
            "ignore_patterns": ["*.md", "*.txt", ".gitattributes"]
        },
        {
            "repo_id": "openai/clip-vit-large-patch14",
            "relative_path": "models/clip/clip-vit-large-patch14",
            "ignore_patterns": ["*.md", "*.txt", "*.msgpack", "*.bin", "*.h5", ".gitattributes"]
        },
        {
            "repo_id": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "relative_path": "models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "ignore_patterns": ["*.md", "*.txt", "*.safetensors", "open_clip_pytorch_model.bin", ".gitattributes",]
        }
    ]

    print(f"Files will be downloaded to root directory: {root_dir}")

    # Download each model
    for model in models:
        # Construct full download path
        download_path = os.path.join(root_dir, model['relative_path'])
        print(f"\nDownloading {model['repo_id']}...")
        print(f"Download path: {download_path}")
        
        try:
            # Ensure target directory exists
            os.makedirs(download_path, exist_ok=True)
            
            # Download the model
            snapshot_download(
                repo_id=model['repo_id'],
                local_dir=download_path,
                ignore_patterns=model['ignore_patterns'],  # Use model-specific ignore patterns
            )
            print(f"{model['repo_id']} download completed!")
        except Exception as e:
            print(f"Error downloading {model['repo_id']}: {str(e)}")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Download FitDiT related model weights')
    parser.add_argument('--dir', '-d', 
                       type=str, 
                       default="./",
                       help='Specify root directory for downloads (defaults to current directory)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Call download function
    download_models(args.dir)

if __name__ == "__main__":
    main()
