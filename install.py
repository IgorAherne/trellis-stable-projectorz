import subprocess
import sys
import os
import time
from typing import Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error
import socket

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass


def is_rtx5000_or_newer():
    """
    Detect if GPU is RTX 5000 series or newer based on compute capability.
    Returns True if compute capability >= 9.0, False otherwise.
    """
    try:
        # Run nvidia-smi to get compute capability
        result = subprocess.run(
            'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader', 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            print("Warning: Could not detect NVIDIA GPU. Defaulting to standard installation.")
            return False
            
        gpu_info = result.stdout.strip().split(',', 1)
        if len(gpu_info) < 2:
            print(f"Unexpected GPU info format: {result.stdout}")
            return False
            
        gpu_name = gpu_info[0].strip()
        # Parse compute capability
        try:
            compute_cap = float(gpu_info[1].strip())
        except ValueError:
            print(f"Could not parse compute capability: {gpu_info[1]}")
            return False
        
        # RTX 5000 series has compute capability 9.0 or higher
        is_rtx5000_plus = compute_cap >= 9.0
        
        print(f"Detected GPU: {gpu_name}")
        print(f"Compute capability: {compute_cap}")
        print(f"RTX 5000 series or newer: {'Yes' if is_rtx5000_plus else 'No'}")
        
        return is_rtx5000_plus
            
    except Exception as e:
        print(f"Error detecting GPU: {str(e)}. Defaulting to standard installation.")
        return False


def check_connectivity(url: str = "https://pytorch.org", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """
    Check internet connectivity and return more detailed error information.
    """
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.gaierror):
            return False, f"DNS resolution failed: {e.reason}"
        elif isinstance(e.reason, socket.timeout):
            return False, "Connection timed out"
        else:
            return False, f"Connection failed: {e.reason}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"

def get_git_env() -> dict:
    """
    Return a copy of the current environment with paths set so that
    pip's internal 'git clone' uses the *portable* Git and correct CA info.
    Adjust the paths below if your layout differs.
    """
    env = os.environ.copy()
    
    # Example location: your portable Git might be in tools/git
    # Adjust if your structure is different
    PORTABLE_GIT_BASE = os.path.join(os.path.dirname(__file__), "..", "tools", "git")
    
    # Prepend the portable Git folders to PATH (like in update.bat)
    # Make sure these subfolders actually exist in your MinGit layout:
    git_paths = [
        os.path.join(PORTABLE_GIT_BASE, "mingw64", "bin"),
        os.path.join(PORTABLE_GIT_BASE, "cmd"),
        os.path.join(PORTABLE_GIT_BASE, "usr", "bin"),
        os.path.join(PORTABLE_GIT_BASE, "mingw64", "libexec", "git-core"),
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = ";".join(git_paths) + ";" + existing_path
    
    # Set SSL cert variables:
    ca_bundle = os.path.join(PORTABLE_GIT_BASE, "mingw64", "etc", "ssl", "certs", "ca-bundle.crt")
    env["GIT_SSL_CAINFO"] = ca_bundle
    env["SSL_CERT_FILE"]  = ca_bundle  # sometimes also recognized
    
    # (Optional) If needed, you can also ensure these:
    # env["GIT_HTTP_LOW_SPEED_LIMIT"] = "1000"
    # env["GIT_HTTP_LOW_SPEED_TIME"]  = "60"
    
    return env

def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES) -> subprocess.CompletedProcess:
    """
    Run a command with retry logic, ensuring we pass an environment
    that forces pip (and git) to use the *portable* Git and correct CA certs.
    """
    last_error = None
    env = get_git_env()  # Build the environment for every command
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
                # Check connectivity before retry
                connected, error_msg = check_connectivity()
                if not connected:
                    print(f"Connection check failed: {error_msg}")
                    print(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
            
            # If pip install is recognized, let's slightly alter the command to ensure no caching, etc.
            if cmd.startswith('pip install'):
                args = cmd[11:]
                cmd = f'"{sys.executable}" -m pip install --no-cache-dir --isolated {args}'
            
            # If installing with pip, we want to show a progress bar
            if "pip install" in cmd:
                if "--progress-bar" not in cmd:
                    cmd += " --progress-bar=on"
                # We pipe stderr but let stdout stream directly (for the progress bar)
                result = subprocess.run(cmd, shell=True, text=True, stdout=sys.stdout, stderr=subprocess.PIPE, env=env)
            else:
                # For all other commands, capture both stdout and stderr
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                return result
            
            last_error = result
            print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output:\n{result.stderr}")
            
        except Exception as e:
            last_error = e
            print(f"\nException during {desc or cmd} (attempt {attempt + 1}/{max_retries}):")
            print(str(e))
        
        if attempt < max_retries - 1:
            print(f"Waiting {RETRY_DELAY} seconds before retry...")
            time.sleep(RETRY_DELAY)
    
    # If we get here, all retries failed
    raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")

def install_dependencies():
    """Install all required dependencies with improved error handling."""
    try:
        # Initial connectivity check
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            print("Please check your connection and try again.")
            sys.exit(1)

        packages_cuda128 = [
            ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128", "Installing PyTorch with CUDA 12.8"),
            ("pip install -r requirements.txt", "Installing basic dependencies"),
            ("pip install -U xformers --index-url https://download.pytorch.org/whl/cu128", "Installing xFormers"),
            ("pip install huggingface_hub", "Installing huggingface_hub"),
            ("pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8", "Installing utils3d"),
        ]
        
        # Local wheel files 
        wheel_files = {
            "cumm_cu128": "whl/cumm_cu128-0.7.13-cp311-cp311-win_amd64.whl",
            "spconv_cu128": "whl/spconv_cu128-2.3.8-cp311-cp311-win_amd64.whl",
            "nvdiffrast": "whl/nvdiffrast-0.3.3-cp311-cp311-win_amd64.whl",
            "diffoctreerast": "whl/diffoctreerast-0.0.0-cp311-cp311-win_amd64.whl",
            "diff_gaussian": "whl/diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl",
            "kaolin": "whl/kaolin-0.17.0-cp311-cp311-win_amd64.whl"
        }

        packages    = packages_cuda128  #commented out, just using cuda 12.8:   if is_rtx5000_or_newer() else packages_cuda124
        
        # Install packages (with retry)
        for cmd, desc in packages:
            run_command_with_retry(cmd, desc)
        
        # Install local wheels
        for name, wheel_path in wheel_files.items():
            wheel = Path(wheel_path)
            if not wheel.exists():
                raise InstallationError(f"Required wheel file not found: {wheel}")
            run_command_with_retry(f"pip install {wheel}", f"Installing {name} from local wheel")
        
        # Install Gradio last
        run_command_with_retry(
            "pip install gradio==4.44.1 gradio_litmodel3d==0.0.1",
            "Installing gradio for web app"
        )
        
        print("\nInstallation completed successfully!")
    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        print("\nSuggestions:")
        print("1. Check your internet connection")
        print("2. Verify your firewall/antivirus isn't blocking connections")
        print("3. Try using a different network if possible")
        print("4. Check if you need to configure a Git or pip proxy")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during installation: {str(e)}")
        sys.exit(1)

def verify_installation():
    """Verify that critical packages were installed correctly."""
    try:
        import torch
        import gradio
        import kaolin
        print(f"PyTorch version: {torch.__version__}")
        print(f"Gradio version: {gradio.__version__}")
        print(f"Kaolin version: {kaolin.__version__}")
        return True
    except ImportError as e:
        print(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    install_dependencies()
    if verify_installation():
        print("\nInstallation completed and verified successfully!")
    else:
        print("\nInstallation completed but verification failed.")
        sys.exit(1)