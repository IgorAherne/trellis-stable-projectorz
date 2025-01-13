import subprocess
import sys
import os
import time
from typing import Optional, Tuple, Union
from pathlib import Path
import urllib.request
import urllib.error
import socket

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass

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
    

def print_cert_debug_info():
    """Print certificate-related debug information"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    tools_dir = os.path.join(parent_dir, 'tools')
    system_dir = os.path.join(parent_dir, 'system')
    
    print("\nCertificate Debug Information:")
    print(f"Script directory: {script_dir}")
    print(f"Parent directory: {parent_dir}")
    
    # Check tools directory
    if os.path.exists(tools_dir):
        print(f"\nContents of tools directory ({tools_dir}):")
        for item in os.listdir(tools_dir):
            print(f"  - {item}")
        
        git_dir = os.path.join(tools_dir, 'git')
        if os.path.exists(git_dir):
            print(f"\nGit directory exists at: {git_dir}")
            cert_path = os.path.join(git_dir, 'mingw64', 'etc', 'ssl', 'certs', 'ca-bundle.crt')
            print(f"Checking ca-bundle.crt: {'EXISTS' if os.path.isfile(cert_path) else 'NOT FOUND'}")
    else:
        print(f"\nWARNING: tools directory not found at {tools_dir}")
        
    # Check system directory
    if os.path.exists(system_dir):
        print(f"\nContents of system directory ({system_dir}):")
        for item in os.listdir(system_dir):
            print(f"  - {item}")
        
        git_dir = os.path.join(system_dir, 'git')
        if os.path.exists(git_dir):
            print(f"\nGit directory exists at: {git_dir}")
            cert_path = os.path.join(git_dir, 'mingw64', 'etc', 'ssl', 'certs', 'ca-bundle.crt')
            print(f"Checking ca-bundle.crt: {'EXISTS' if os.path.isfile(cert_path) else 'NOT FOUND'}")
    else:
        print(f"\nWARNING: system directory not found at {system_dir}")


def check_cert_error(error_text: str) -> bool:
    """Check if error is certificate related"""
    cert_indicators = [
        'ssl.SSLError',
        'ASN1',
        'failed to open CA file',
        'schannel',
        'certificate',
        'cert'
    ]
    return any(indicator.lower() in error_text.lower() for indicator in cert_indicators)


def get_git_env() -> dict:
    """
    Return a copy of the current environment with paths set so that
    pip's internal 'git clone' uses the *portable* Git and correct structure.
    Also sets GIT_SSL_CAINFO so that 'git+https://...' installs succeed.
    """
    env = os.environ.copy()

    # If user has leftover references to old "system\git", forcibly remove them:
    if "GIT_SSL_CAINFO" in env and "system\\git" in env["GIT_SSL_CAINFO"].lower():  del env["GIT_SSL_CAINFO"]
    if "SSL_CERT_FILE"  in env and "system\\git" in env["SSL_CERT_FILE"].lower():  del env["SSL_CERT_FILE"]
    
    PORTABLE_GIT_BASE = os.path.join(os.path.dirname(__file__), "..", "tools", "git")
    
    # Prepend the portable Git folders to PATH
    git_paths = [
        os.path.join(PORTABLE_GIT_BASE, "mingw64", "bin"),
        os.path.join(PORTABLE_GIT_BASE, "cmd"),
        os.path.join(PORTABLE_GIT_BASE, "usr", "bin"),
        os.path.join(PORTABLE_GIT_BASE, "mingw64", "libexec", "git-core"),
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = ";".join(git_paths) + ";" + existing_path
    
    # Reintroduce CA certificate environment variables (if the file actually exists).
    ca_bundle = os.path.join(PORTABLE_GIT_BASE, "mingw64", "etc", "ssl", "certs", "ca-bundle.crt")
    print('')
    print('setting the path to the ca_bundle: ' + ca_bundle + f'   path exists: { os.path.isfile(ca_bundle) }')
    print('')
    if os.path.isfile(ca_bundle):
        env["GIT_SSL_CAINFO"] = ca_bundle
        env["SSL_CERT_FILE"]  = ca_bundle
    
    return env



def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES) -> subprocess.CompletedProcess:
    """
    Run a command with retry logic, ensuring we pass an environment
    that forces pip (and git) to use the *portable* Git (via PATH).
    """
    last_error = None

    # Print Git environment info only if this is a git-related command
    if 'git' in cmd.lower():
        env = get_git_env()
        print(f"\nGit executable path: {env.get('PATH', '').split(';')[0]}")
        print(f"Git SSL cert path: {env.get('GIT_SSL_CAINFO', 'Not set')}")

    for attempt in range(max_retries):
        # Attempt to run the command, storing the result or exception
        attempt_result = _attempt_command_once(cmd, desc, attempt, max_retries)
        
        # If we got a CompletedProcess, check its return code
        if isinstance(attempt_result, subprocess.CompletedProcess):
            if attempt_result.returncode == 0:
                return attempt_result  # Success
            last_error = attempt_result  # A subprocess.CompletedProcess that failed
            error_text = attempt_result.stderr if hasattr(attempt_result, 'stderr') and attempt_result.stderr else str(attempt_result)
        else:
            # It's an exception, capture it
            last_error = attempt_result
            error_text = str(attempt_result)

        # Print any error info
        print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
        print(f"Error output:\n{error_text}")
        
        # Check for certificate issues
        if check_cert_error(error_text):
            print_cert_debug_info()

        # If we have more attempts left, wait
        if attempt < max_retries - 1:
            print(f"Waiting {RETRY_DELAY} seconds before retry...")
            time.sleep(RETRY_DELAY)

    # If we get here, all retries failed
    raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")


def _attempt_command_once(cmd:str, desc:Optional[str], attempt:int, max_retries:int) -> Union[subprocess.CompletedProcess, Exception]:
    """
    Single attempt at running a command. This is extracted from the main for-loop body in
    run_command_with_retry to separate the logic of a single execution attempt.
    """
    env = get_git_env()  # build fresh env each attempt

    try:
        # If this is a retry (i.e., not the first attempt), do connectivity checks, etc.
        if attempt > 0:
            print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
            # Check connectivity before retry
            connected, error_msg = check_connectivity()
            if not connected:
                print(f"Connection check failed: {error_msg}")
                if check_cert_error(error_msg):
                    print_cert_debug_info()
                print(f"Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)

        # If pip install is recognized, alter the command for no-cache, isolation, and verbosity
        if cmd.startswith('pip install'):
            args = cmd[11:]
            python_path = str(Path(sys.executable).resolve())  # get the absolute path to Python
            cmd = f'"{python_path}" -m pip install --no-cache-dir --isolated {args} --verbose'
            print(f"\nExecuting pip install command: {cmd}")

        # If installing with pip, we want to show a progress bar
        if "pip install" in cmd:
            if "--progress-bar" not in cmd:
                cmd += " --progress-bar=on"
            # We pipe stderr but let stdout stream directly (for the progress bar)
            result = subprocess.run(cmd, shell=True, text=True, stdout=sys.stdout, stderr=subprocess.PIPE, env=env)
        else:
            # For all other commands, capture both stdout and stderr
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

        return result

    except Exception as e:
        # Return the exception so the parent function can handle it
        return e


def install_dependencies():
    """Install all required dependencies with improved error handling."""
    try:
        # Initial connectivity check
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            print("Please check your connection and try again.")
            sys.exit(1)
        
        # List of packages to install with pip. USE THE "" AROUND PACKAGES, to avoid issues with > or < characters
        packages = [
            ('pip install -r "requirements.txt"', 'Installing basic dependencies'),
            ('pip install "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2" --index-url "https://download.pytorch.org/whl/cu118"', 'Installing PyTorch 2.1.2 with CUDA 11.8'),
            ('pip install "xformers==0.0.23.post1" --index-url "https://download.pytorch.org/whl/cu118"', 'Installing xformers'),
            ('pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"', 'Installing utils3d'),
            ('pip install "kaolin==0.17.0" -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html"', 'Installing Kaolin'),
            ('pip install "spconv-cu118==2.3.6"', 'Installing spconv'),
            
            # Modified numpy installation command with consistent quotation
            ('pip install "numpy>=1.22.0,<2.0.0"', 'Installing numpy with version constraints'),
            ('pip install "plyfile"', 'Installing plyfile for reading .ply data'),
        ]
        
        # Local wheel files
        wheel_files = {
            'nvdiffrast': 'whl/nvdiffrast-0.3.3-cp311-cp311-win_amd64.whl',
            'diffoctreerast': 'whl/diffoctreerast-0.0.0-cp311-cp311-win_amd64.whl',
            'diff_gaussian': 'whl/diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl'
        }
        
        # Install packages (with retry)
        for cmd, desc in packages:
            run_command_with_retry(cmd, desc)
        
        # Install local wheels
        for name, path_str in wheel_files.items():
            path = Path(path_str)
            if not path.exists():
                raise InstallationError(f"Required wheel file not found: {path}")
            run_command_with_retry(f'pip install "{path_str}"', f'Installing {name} from local wheel')
        
        # Install Gradio last
        run_command_with_retry(
            'pip install "gradio==4.44.1" "gradio_litmodel3d==0.0.1"',
            'Installing gradio for web app'
        )
        
        print("\nInstallation completed successfully!  Launching, please wait...")
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
        import plyfile
        print(f"PyTorch version: {torch.__version__}")
        print(f"Gradio version: {gradio.__version__}")
        print(f"Kaolin version: {kaolin.__version__}")
        print("plyfile imported successfully.")
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
