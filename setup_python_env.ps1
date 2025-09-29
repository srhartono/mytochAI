# This script sets up Python environment paths
# Run this script with administrator privileges

# Get Python installation directory
$pythonDir = Split-Path -Parent (Get-Command python -ErrorAction SilentlyContinue).Path
# $venvDir = "C:\Users\thatc\.vscode\projects and stuff in vscode\uhhhh\.venv\Scripts"
$venvDir = Get-Location

# Function to add a path to environment variable if it doesn't exist
function Add-ToPath {
    param(
        [string]$PathToAdd,
        [string]$PathType # 'Machine' for system-wide or 'User' for user-specific
    )
    
    if (-not [string]::IsNullOrEmpty($PathToAdd)) {
        $currentPath = [Environment]::GetEnvironmentVariable('Path', $PathType)
        if ($currentPath -notlike "*$PathToAdd*") {
            $newPath = "$currentPath;$PathToAdd"
            [Environment]::SetEnvironmentVariable('Path', $newPath, $PathType)
            Write-Host "Added $PathToAdd to $PathType Path"
        } else {
            Write-Host "$PathToAdd already exists in $PathType Path"
        }
    }
}

# Add Python paths
if ($pythonDir) {
    Add-ToPath -PathToAdd $pythonDir -PathType 'User'
    Add-ToPath -PathToAdd "$pythonDir\Scripts" -PathType 'User'
} else {
    Write-Host "Python installation not found in PATH. Please ensure Python is installed."
}

# Add Virtual Environment paths
if (Test-Path $venvDir) {
    Add-ToPath -PathToAdd $venvDir -PathType 'User'
    Write-Host "Virtual environment paths added successfully"
} else {
    Write-Host "Virtual environment not found at $venvDir"
}

# Set PYTHONPATH if needed
$pythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'User')
if (-not $pythonPath) {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $env:USERPROFILE, 'User')
    Write-Host "Set PYTHONPATH to user home directory"
}

Write-Host "`nEnvironment setup complete. Please restart your terminal or VS Code for changes to take effect."

# Display current Python-related paths
Write-Host "`nCurrent Python Configuration:"
Write-Host "------------------------"
Write-Host "Python Directory: $pythonDir"
Write-Host "Virtual Environment: $venvDir"
Write-Host "PYTHONPATH: $([Environment]::GetEnvironmentVariable('PYTHONPATH', 'User'))"
Write-Host "------------------------"

# Verify installations
try {
    $pythonVersion = python --version
    Write-Host "Python Version: $pythonVersion"
} catch {
    Write-Host "Could not verify Python installation"
}

try {
    $pipVersion = pip --version
    Write-Host "Pip Version: $pipVersion"
} catch {
    Write-Host "Could not verify pip installation"
}