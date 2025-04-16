# Day - 9 Python Project Setup and Best Practices

## Overview
This lesson covers essential project setup practices in Python, including virtual environments, dependency management, and version control with GitHub. You'll learn how to structure projects professionally and manage them effectively.

## Learning Objectives
- Master virtual environment creation and management
- Understand dependency management with pip
- Learn Git basics and GitHub workflow
- Create professional project structures
- Implement best practices for Python projects

## Prerequisites
- Understanding of Python fundamentals (variables, functions, OOP)
- Knowledge of modules and packages
- Python 3.x installed
- Basic command line knowledge
- Completion of [07 - Testing and Debugging](07-Python-Testing-Debugging.md)
- Completion of [08 - Functional Programming](08-Python-Functional-Programming.md)
- Python 3.x installed on your computer
- Git installed on your computer

## Time Estimate
- Reading: 35 minutes
- Practice: 55 minutes
- Project Setup: 45 minutes

---

## 1. Virtual Environments

### Understanding Virtual Environments
Virtual environments are isolated Python environments that allow you to:
- Install packages specific to a project
- Avoid conflicts between project dependencies
- Maintain clean and reproducible development environments

### Creating and Managing Virtual Environments
```bash
# Create a new virtual environment
python -m venv myproject_env

# Activate the virtual environment
# On Windows:
myproject_env\\Scripts\\activate
# On macOS/Linux:
source myproject_env/bin/activate

# Verify activation
python -c "import sys; print(sys.prefix)"

# Deactivate when done
deactivate
```

### Best Practices for Virtual Environments
1. **Naming Conventions**
   ```bash
   # Project-specific name
   python -m venv projectname_env
   
   # Generic name (when in project directory)
   python -m venv venv
   ```

2. **Directory Structure**
   ```plaintext
   myproject/
   ├── myproject_env/    # Virtual environment
   ├── src/              # Source code
   ├── tests/            # Test files
   ├── requirements.txt  # Dependencies
   └── README.md        # Documentation
   ```

3. **Environment Variables**
   ```bash
   # Create .env file
   touch .env
   
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

### Real-world Example: Project Setup
```python
from pathlib import Path
import subprocess
import os
import sys

class ProjectSetup:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.base_dir = Path(project_name)
        self.venv_dir = self.base_dir / "venv"
    
    def create_directory_structure(self):
        """Create project directory structure."""
        # Create main directories
        dirs = [
            self.base_dir,
            self.base_dir / "src",
            self.base_dir / "tests",
            self.base_dir / "docs"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def create_virtual_environment(self):
        """Create and activate virtual environment."""
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_dir)],
                check=True
            )
            print(f"Created virtual environment: {self.venv_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    
    def create_initial_files(self):
        """Create initial project files."""
        files = {
            "README.md": f"# {self.project_name}\n\nProject description goes here.",
            ".gitignore": "venv/\n__pycache__/\n*.pyc\n.env\n",
            "requirements.txt": "# Project dependencies\n",
            "src/__init__.py": "",
            "tests/__init__.py": "",
        }
        
        for file_path, content in files.items():
            full_path = self.base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            print(f"Created file: {full_path}")

# Example usage
def setup_new_project():
    # Initialize project
    setup = ProjectSetup("my_awesome_project")
    
    # Create structure
    setup.create_directory_structure()
    
    # Create virtual environment
    setup.create_virtual_environment()
    
    # Create initial files
    setup.create_initial_files()
    
    print("\nProject setup complete! Next steps:")
    print("1. cd my_awesome_project")
    print("2. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   .\\venv\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        print("   source venv/bin/activate")
    print("3. Install required packages:")
    print("   pip install -r requirements.txt")

# Run setup
if __name__ == "__main__":
    setup_new_project()
```

## 2. Dependency Management

### Understanding Package Management
Package management in Python involves:
- Installing required libraries
- Managing versions
- Tracking dependencies
- Ensuring reproducibility

### Using pip
```bash
# Install a package
pip install requests

# Install specific version
pip install requests==2.28.1

# Install with version constraints
pip install 'requests>=2.28.0,<3.0.0'

# List installed packages
pip list

# Show package details
pip show requests
```

### Managing Requirements
```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt
```

### Best Practices for Dependencies
1. **Version Pinning**
   ```txt
   # requirements.txt
   requests==2.28.1
   pandas==1.5.3
   python-dotenv==0.21.1
   ```

2. **Development Dependencies**
   ```txt
   # requirements-dev.txt
   -r requirements.txt
   pytest==7.3.1
   black==23.3.0
   flake8==6.0.0
   ```

3. **Using setup.py**
   ```python
   from setuptools import setup, find_packages

   setup(
       name="my_awesome_project",
       version="0.1.0",
       packages=find_packages(),
       install_requires=[
           'requests>=2.28.0,<3.0.0',
           'pandas>=1.5.0,<2.0.0',
       ],
       extras_require={
           'dev': [
               'pytest>=7.3.0',
               'black>=23.3.0',
               'flake8>=6.0.0',
           ],
       },
   )
   ```

### Real-world Example: Package Manager
```python
from pathlib import Path
import subprocess
import json
from typing import Dict, List, Optional

class PackageManager:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.requirements_file = self.project_dir / "requirements.txt"
        self.dev_requirements_file = self.project_dir / "requirements-dev.txt"
    
    def install_package(self, package: str, version: Optional[str] = None):
        """Install a Python package."""
        cmd = ["pip", "install"]
        if version:
            cmd.append(f"{package}=={version}")
        else:
            cmd.append(package)
        
        subprocess.run(cmd, check=True)
        print(f"Installed {package}" + (f" version {version}" if version else ""))
    
    def save_requirements(self, include_dev: bool = False):
        """Save installed packages to requirements files."""
        # Get installed packages
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        requirements = result.stdout
        
        # Split into main and dev requirements
        main_deps = []
        dev_deps = []
        
        for line in requirements.splitlines():
            if any(dev_pkg in line.lower() for dev_pkg in 
                  ['pytest', 'black', 'flake8', 'mypy']):
                dev_deps.append(line)
            else:
                main_deps.append(line)
        
        # Save main requirements
        self.requirements_file.write_text('\n'.join(main_deps))
        print(f"Saved requirements to {self.requirements_file}")
        
        # Save dev requirements if requested
        if include_dev:
            dev_content = ["-r requirements.txt", *dev_deps]
            self.dev_requirements_file.write_text('\n'.join(dev_content))
            print(f"Saved dev requirements to {self.dev_requirements_file}")
    
    def install_requirements(self, dev: bool = False):
        """Install packages from requirements files."""
        if dev and self.dev_requirements_file.exists():
            subprocess.run(
                ["pip", "install", "-r", str(self.dev_requirements_file)],
                check=True
            )
            print("Installed development requirements")
        elif self.requirements_file.exists():
            subprocess.run(
                ["pip", "install", "-r", str(self.requirements_file)],
                check=True
            )
            print("Installed production requirements")
        else:
            print("No requirements file found")

# Example usage
def manage_project_dependencies():
    # Initialize package manager
    pkg_manager = PackageManager("my_awesome_project")
    
    # Install some packages
    pkg_manager.install_package("requests", "2.28.1")
    pkg_manager.install_package("pandas", "1.5.3")
    
    # Install dev packages
    pkg_manager.install_package("pytest", "7.3.1")
    pkg_manager.install_package("black", "23.3.0")
    
    # Save requirements
    pkg_manager.save_requirements(include_dev=True)
    
    # Install from requirements
    pkg_manager.install_requirements(dev=True)
```
