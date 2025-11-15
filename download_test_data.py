#!/usr/bin/env python3
"""
Download test data from HuggingFace dataset repository.
Only down data/evalloads files under data/eval/ directory.
"""

import os
import sys
from huggingface_hub import snapshot_download, HfApi

# Set HuggingFace endpoint to use mirror


def main():
    repo_id = 'qingfei1/R-Search_datasets'
    local_dir = './test_data'
    
    print(f"Setting HF_ENDPOINT to: {os.environ.get('HF_ENDPOINT')}")
    print(f"Connecting to repository: {repo_id}")
    print(f"Target directory: {local_dir}")
    print(f"Download pattern: data/eval/**")
    print("-" * 50)
    
    try:
        # First, verify the repository exists and list files
        print("Checking repository accessibility...")
        api = HfApi()
        try:
            files = api.list_files_info(repo_id=repo_id, repo_type='dataset')
            print(f"Repository accessible. Found {len(files)} files total.")
            eval_files = [f for f in files if 'data/eval/' in f.path]
            print(f"Files matching 'data/eval/': {len(eval_files)}")
            if eval_files:
                print("Sample files to download:")
                for f in eval_files[:5]:
                    print(f"  - {f.path} ({f.size / 1024 / 1024:.2f} MB)")
                if len(eval_files) > 5:
                    print(f"  ... and {len(eval_files) - 5} more files")
            else:
                print("WARNING: No files found matching 'data/eval/' pattern!")
                print("Available directories:")
                dirs = set([f.path.split('/')[0] for f in files[:20]])
                for d in sorted(dirs):
                    print(f"  - {d}/")
        except Exception as e:
            print(f"Warning: Could not list repository files: {e}")
            print("Continuing with download attempt...")
        
        print("-" * 50)
        print("Starting download...")
        
        # Download with progress
        snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            local_dir=local_dir,
            allow_patterns=['data/eval/**'],
            local_dir_use_symlinks=False,
        )
        
        print("-" * 50)
        print("Download completed successfully!")
        print(f"Files saved to: {os.path.abspath(local_dir)}")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during download: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()