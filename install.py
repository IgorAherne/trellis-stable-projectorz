import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, desc=None):
    """Run a command and handle errors"""
    if desc:
        print(f"\n{desc}...")
    
    # Replace 'pip install' with explicit python -m pip install
    if cmd.startswith('pip install'):
        # Extract everything after 'pip install'
        args = cmd[11:]  # 11 is the length of 'pip install '
        cmd = f"{sys.executable} -m pip install {args}"
    
    # Special handling for pip install commands to show progress
    if "pip install" in cmd:
        # Add --progress-bar=on and remove capture_output to show live progress
        if "--progress-bar" not in cmd:
            cmd += " --progress-bar=on"
        result = subprocess.run(cmd, shell=True, text=True)
    else:
        # For non-pip commands, capture output for logging
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\nError during {desc or cmd}:")
        if hasattr(result, 'stderr'):  # Only if we captured output
            print("\nSTDERR:")
            print(result.stderr)
        print("\nCommand that failed:")
        print(cmd)
        sys.exit(1)
    
    # Print stdout for successful commands, but only if we captured it
    if hasattr(result, 'stdout') and result.stdout:
        print(result.stdout)
    
    return result


def check_internet():
    try:
        import urllib.request
        urllib.request.urlopen('https://pytorch.org', timeout=5)
        return True
    except:
        return False

def install_dependencies():
    if not check_internet():
        print("Error: No internet connection detected. Please check your connection and try again.")
        sys.exit(1)
        
    # Basic requirements first
    run_command("pip install -r requirements.txt", "Installing basic dependencies, please wait a 5-15 minutes...")
    
    # PyTorch with CUDA 11.8
    print("\nDownloading PyTorch packages (this is a large download, approximately 2.7GB)...")
    print("This might take several minutes depending on your internet connection.")
    run_command(
        f"pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --progress-bar on",
        "Installing PyTorch 2.1.2 with CUDA 11.8"
    )
    
    # xformers
    run_command(
        "pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118",
        "Installing xformers"
    )
    
    # utils3d
    run_command(
        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        "Installing utils3d"
    )

    # Kaolin
    run_command(
        "pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html",
        "Installing Kaolin"
    )
    
    # spconv
    run_command(
        "pip install spconv-cu118==2.3.6",
        "Installing spconv"
    )
    
    # nvdiffrast - install from local wheel file
    nvdiffrast_wheel = Path("whl/nvdiffrast-0.3.3-cp311-cp311-win_amd64.whl")
    if not nvdiffrast_wheel.exists():
        print(f"Error: Required wheel file not found: {nvdiffrast_wheel}")
        sys.exit(1)
    run_command(
        f"pip install {nvdiffrast_wheel}",
        "Installing nvdiffrast from local wheel"
    )

     # diffoctreerast - install from local wheel file
    diffoctreerast_wheel = Path("whl/diffoctreerast-0.0.0-cp311-cp311-win_amd64.whl")
    if not diffoctreerast_wheel.exists():
        print(f"Error: Required wheel file not found: {diffoctreerast_wheel}")
        sys.exit(1)
    run_command(
        f"pip install {diffoctreerast_wheel}",
        "Installing diffoctreerast from local wheel"
    )

    # diff gaussian raster - install from local wheel file
    diff_gauss_wheel = Path("whl/diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl")
    if not diff_gauss_wheel.exists():
        print(f"Error: Required wheel file not found: {diff_gauss_wheel}")
        sys.exit(1)
    run_command(
        f"pip install {diff_gauss_wheel}",
        "Installing diff_gaussian_rasterization from local wheel"
    )

    # install gradio for web app
    run_command(
        f"pip install gradio==4.44.1 gradio_litmodel3d==0.0.1",
        "Installing gradio for web app"
    )


if __name__ == "__main__":
    install_dependencies()
    print("\nInstallation completed successfully!")