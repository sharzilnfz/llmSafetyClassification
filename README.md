# LLM Safety Classification (CSE440 NLP Project)

Multi-Class Text Classification for LLM Safety using **[UV](https://github.com/astral-sh/uv)**.

---

## ⚡ Quick Setup for Team Members (3 Steps)

Because this project uses `uv` with a committed `uv.lock`, your entire environment, Python version (3.13), and dependencies (PyTorch, Transformers, Scikit-Learn, Gensim, etc.) will install in seconds with zero version conflicts across macOS, Windows, and Linux.

### Step 1: Install `uv` (if not already installed)

- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or on macOS with Homebrew:
  brew install uv
  ```
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

---

### Step 2: Clone Repo & Sync Environment

```bash
# 1. Clone the repository
git clone https://github.com/sharzilnfz/llmSafetyClassification.git
cd llmSafetyClassification

# 2. Automatically download Python 3.13 and install all dependencies
uv sync
```

---

### Step 3: Open the Notebook

1. Open the project folder in **VS Code** or **Cursor**.
2. Open [`cse440_project.ipynb`](cse440_project.ipynb).
3. In the top right corner, click **Select Kernel** -> **Python Environments...** -> Choose the `.venv` environment (or `Python 3.13 (.venv)`).
4. Run your cells!

*(Optional: If you prefer JupyterLab in the browser, simply run `uv run jupyter lab`)*.

---

## 💡 Adding New Packages

If you need to install a new library, run:
```bash
uv add <package_name>
```
`uv` will automatically update `pyproject.toml` and `uv.lock`. Commit those files so everyone on the team stays in sync!
