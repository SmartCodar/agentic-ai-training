# Python Setup: Virtual Environments with `venv` and Miniconda

## Overview
Proper Python environment setup is crucial for managing dependencies and avoiding conflicts across projects. This guide covers setting up isolated environments using built-in `venv` and Miniconda, along with CLI commands and application examples.

## 🌟 Learning Objectives
By the end of this session, you will:
- Understand the purpose of virtual environments
- Set up environments using `venv` and Miniconda
- Run isolated applications in each environment
- Install and manage packages per project

## 📋 Prerequisites
- Python 3.11+ installed
- Miniconda downloaded ([link](https://docs.conda.io/en/latest/miniconda.html))
- Basic familiarity with terminal/command line

---

## 1. Using Python's Built-in `venv`

### Step 1: Create Environment
```bash
python3 -m venv venv
```

### Step 2: Activate Environment
- **Linux/macOS:**
```bash
source venv/bin/activate
```
- **Windows:**
```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install fastapi uvicorn
```

### Step 4: Run Example App
```python
# app.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from venv!"}
```

```bash
uvicorn app:app --reload
```

### Step 5: Deactivate
```bash
deactivate
```

---

## 2. Using Miniconda (Recommended for Data Science)

### Step 1: Create Conda Environment
```bash
conda create -n myenv python=3.11
```

### Step 2: Activate Environment
```bash
conda activate myenv
```

### Step 3: Install Dependencies
```bash
conda install fastapi uvicorn -c conda-forge
```

### Step 4: Run App in Conda Environment
```python
# app.py (same as above)
```
```bash
uvicorn app:app --reload
```

### Step 5: Deactivate
```bash
conda deactivate
```

---

## 3. Managing Environments

### Export Requirements (venv)
```bash
pip freeze > requirements.txt
```
### Restore
```bash
pip install -r requirements.txt
```

### Export Conda Env
```bash
conda env export > environment.yml
```
### Restore Conda Env
```bash
conda env create -f environment.yml
```

---

## ✅ Summary
- `venv` is lightweight and works out-of-the-box
- Conda is powerful for scientific projects with native support for non-Python dependencies
- Both allow isolated dependency management and fast switching

---

## 🔍 Common Issues and Fixes
| Issue | Fix |
|-------|------|
| `ModuleNotFoundError` | Re-activate env or reinstall packages |
| `conda activate` not found | Run `conda init` and restart terminal |
| `Permission denied` (venv) | Use `python3 -m venv --clear venv` |

## 📚 Additional Resources
- [Python venv docs](https://docs.python.org/3/library/venv.html)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [FastAPI Quickstart](https://fastapi.tiangolo.com/)

## ✅ Knowledge Check
1. When should you prefer `conda` over `venv`?
2. How do you activate a `venv` on Windows?
3. What is `environment.yml` used for?
4. How do you list installed packages in `venv` and Conda?

---

> **Navigation**
> - [← Previous: Role-Based Prompting](20-Python-Role-Based-Prompting.md)
> - [Next: Python CLI Apps →](DayX-Python-CLI-Apps.md)

