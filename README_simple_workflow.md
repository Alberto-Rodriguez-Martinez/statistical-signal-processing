Quick student instructions (minimal steps)

Repository: https://github.com/Alberto-Rodriguez-Martinez/statistical-signal-processing (branch: main)

Where to read the helper code
- Browse the folder `codigo/` on GitHub:
  https://github.com/Alberto-Rodriguez-Martinez/statistical-signal-processing/tree/main/codigo
  — this always shows the latest code the instructor pushed.

What I (instructor) did
- Each notebook should include a small top code cell that:
  - makes `codigo` importable in Binder and local sessions, and
  - in Google Colab it automatically fetches the latest `codigo/` from GitHub for the session.
- This keeps things simple for students: they don't need to use git or installs to read/run the helper code.

How to run notebooks (pick one)

1) Binder (recommended for live class, zero installs)
- Click the Binder badge in the repo README (or use the Badge link provided in the repo).
- Open a notebook and run cells. The first cell ensures `codigo` is importable.
- New Binder sessions pick up the newest code automatically after I push changes.

2) Google Colab (recommended if you want to save to Drive)
- Open Colab and use the GitHub tab, or use this link to browse the repo in Colab:
  https://colab.research.google.com/github/Alberto-Rodriguez-Martinez/statistical-signal-processing
- If a notebook is opened in Colab, run the first code cell (it will fetch the latest `codigo/` automatically).
- If you prefer, run this install cell at the top to get codigo as a pip package in the session:
```python
# (run in Colab to install/refresh codigo for the session)
!pip install --upgrade git+https://github.com/Alberto-Rodriguez-Martinez/statistical-signal-processing.git@main#egg=course&subdirectory=codigo
```
- After the install, you may need to restart the runtime or re-import modules.

3) Local (optional - for students who like working locally)
- Clone the repo and open notebooks in JupyterLab/Notebook:
```bash
git clone https://github.com/Alberto-Rodriguez-Martinez/statistical-signal-processing.git
cd statistical-signal-processing/notebooks    # if your notebooks are in notebooks/
# open JupyterLab/Notebook as you normally do
jupyter lab
```
- The top-of-notebook cell will add the repo root to Python's path so `import codigo` works.
- If you want to keep a local copy updated, run:
```bash
git pull origin main
```

Notes for students
- To *see* the code: open `codigo/` on GitHub (easy, no git knowledge required).
- To *run* the notebooks: use Binder or Colab — no installs required.
- If I update `codigo/`, new Binder sessions and re-running the top Colab cell will pick up the latest code.
- If anything fails, re-run the top cell, restart the runtime/kernel, or ask in class / open an issue in the repo.

Instructor tip (for you)
- Add the top-of-notebook snippet (provided) as the first code cell in each notebook, or keep a small helper notebook students run first.