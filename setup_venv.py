#!/usr/bin/env python
"""
Setup script to create and configure virtual environment for the genomics framework.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True,
            shell=True if isinstance(cmd, str) else False
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            raise
        return e

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version < (3, 8):
        print(f"Error: Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.8 or newer.")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_venv(project_root):
    """Create virtual environment."""
    venv_path = project_root / ".venv"
    
    if venv_path.exists():
        print("Virtual environment already exists.")
        response = input("Do you want to recreate it? (y/N): ").strip().lower()
        if response == 'y':
            print("Removing existing virtual environment...")
            import shutil
            try:
                shutil.rmtree(venv_path)
                print("✓ Existing virtual environment removed")
            except Exception as e:
                print(f"Warning: Could not fully remove existing venv: {e}")
        else:
            print("Using existing virtual environment.")
            return venv_path
    
    print("Creating virtual environment...")
    run_command([sys.executable, "-m", "venv", str(venv_path)], cwd=project_root)
    
    if not venv_path.exists():
        print("Error: Failed to create virtual environment")
        return None
    
    print(f"✓ Virtual environment created at {venv_path}")
    return venv_path

def get_activation_command(venv_path):
    """Get the appropriate activation command for the platform."""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "Activate.ps1")
    else:
        return f"source {venv_path / 'bin' / 'activate'}"

def get_pip_executable(venv_path):
    """Get the pip executable path for the virtual environment."""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "pip.exe")
    else:
        return str(venv_path / "bin" / "pip")

def install_requirements(venv_path, project_root):
    """Install requirements in the virtual environment."""
    pip_exe = get_pip_executable(venv_path)
    python_exe = pip_exe.replace("pip", "python")
    if platform.system() == "Windows":
        python_exe = pip_exe.replace("pip.exe", "python.exe")
    
    # Upgrade pip first using python -m pip
    print("Upgrading pip...")
    try:
        run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    except Exception as e:
        print(f"Warning: Could not upgrade pip: {e}")
        print("Continuing with existing pip version...")
    
    # Install basic requirements
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        print("Installing core requirements...")
        run_command([pip_exe, "install", "-r", str(requirements_file)])
    else:
        print("No requirements.txt found, installing basic packages...")
        basic_packages = [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "pyyaml>=6.0"
        ]
        run_command([pip_exe, "install"] + basic_packages)
    
    # Install the framework in development mode
    print("Installing framework in development mode...")
    run_command([pip_exe, "install", "-e", "."], cwd=project_root)
    
    print("✓ Core requirements installed successfully")

def install_optional_dependencies(venv_path, project_root):
    """Install optional dependencies based on user choice."""
    pip_exe = get_pip_executable(venv_path)
    
    print("\nOptional dependencies:")
    print("1. PyTorch (CPU-only version)")
    print("2. PyTorch with CUDA support") 
    print("3. Advanced visualization (bokeh, plotly)")
    print("4. Skip optional dependencies")
    
    choice = input("Choose an option (1-4): ").strip()
    
    if choice == "1":
        print("Installing PyTorch (CPU-only)...")
        try:
            run_command([pip_exe, "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
            print("✓ PyTorch CPU installed successfully")
        except Exception as e:
            print(f"Warning: Failed to install PyTorch: {e}")
    elif choice == "2":
        print("Installing PyTorch with CUDA 11.8 support...")
        try:
            run_command([pip_exe, "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118"])
            print("✓ PyTorch CUDA installed successfully")
        except Exception as e:
            print(f"Warning: Failed to install PyTorch CUDA: {e}")
    elif choice == "3":
        print("Installing advanced visualization...")
        try:
            run_command([pip_exe, "install", "bokeh>=2.4.0", "pingouin>=0.5.0"])
            print("✓ Advanced visualization installed successfully")
        except Exception as e:
            print(f"Warning: Failed to install some packages: {e}")
    else:
        print("Skipping optional dependencies")
    
    print("\nNote: For bioinformatics tools (pysam, pybedtools), please install manually:")
    print("  conda install -c bioconda pysam pybedtools  # Recommended")
    print("  OR see requirements-optional.txt for manual installation")

def create_activation_scripts(venv_path, project_root):
    """Create convenient activation scripts."""
    
    # Windows batch file
    if platform.system() == "Windows":
        batch_content = f"""@echo off
echo Activating genomics framework virtual environment...
call "{venv_path}\\Scripts\\activate.bat"
echo Virtual environment activated!
echo Run 'python main.py --demo' to test the framework.
cmd /k
"""
        batch_file = project_root / "activate_env.bat"
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        print(f"✓ Created activation script: {batch_file}")
        
        # PowerShell script
        ps_content = f"""Write-Host "Activating genomics framework virtual environment..." -ForegroundColor Green
& "{venv_path}\\Scripts\\Activate.ps1"
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host "Run 'python main.py --demo' to test the framework." -ForegroundColor Yellow
"""
        ps_file = project_root / "activate_env.ps1"
        with open(ps_file, 'w') as f:
            f.write(ps_content)
        print(f"✓ Created PowerShell activation script: {ps_file}")
    
    # Bash script for Linux/macOS
    else:
        bash_content = f"""#!/bin/bash
echo "Activating genomics framework virtual environment..."
source "{venv_path}/bin/activate"
echo "Virtual environment activated!"
echo "Run 'python main.py --demo' to test the framework."
exec "$SHELL"
"""
        bash_file = project_root / "activate_env.sh"
        with open(bash_file, 'w') as f:
            f.write(bash_content)
        
        # Make executable
        os.chmod(bash_file, 0o755)
        print(f"✓ Created activation script: {bash_file}")

def verify_installation(venv_path, project_root):
    """Verify the installation works."""
    print("\nVerifying installation...")
    
    python_exe = get_pip_executable(venv_path).replace("pip", "python")
    if platform.system() == "Windows":
        python_exe = python_exe.replace("pip.exe", "python.exe")
    
    # Test import
    test_script = """
try:
    import sys
    sys.path.insert(0, '.')
    from src import DataLoader, GenomicsPipeline
    print("✓ Framework imports successful")
    print("✓ Installation verified!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
"""
    
    result = run_command([python_exe, "-c", test_script], cwd=project_root, check=False)
    
    if result.returncode == 0:
        print("✓ Installation verification passed")
        return True
    else:
        print("✗ Installation verification failed")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("Genomics Correlation Framework - Virtual Environment Setup")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    venv_path = create_venv(project_root)
    if not venv_path:
        return 1
    
    try:
        # Install requirements
        install_requirements(venv_path, project_root)
        
        # Install optional dependencies
        install_optional_dependencies(venv_path, project_root)
        
        # Create activation scripts
        create_activation_scripts(venv_path, project_root)
        
        # Verify installation
        if verify_installation(venv_path, project_root):
            print("\n" + "=" * 60)
            print("✓ SETUP COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            activation_cmd = get_activation_command(venv_path)
            
            print(f"\nTo activate the environment:")
            if platform.system() == "Windows":
                print(f"  PowerShell: .\\activate_env.ps1")
                print(f"  Command Prompt: activate_env.bat")
                print(f"  Manual: {activation_cmd}")
            else:
                print(f"  ./activate_env.sh")
                print(f"  Manual: {activation_cmd}")
            
            print(f"\nTo test the framework:")
            print(f"  python main.py --demo")
            
            return 0
        else:
            print("\n✗ Setup completed with errors")
            return 1
            
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)