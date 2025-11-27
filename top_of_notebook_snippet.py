# Top-of-notebook helper (paste into the first code cell of each notebook)
# - makes codigo importable in local/Binder sessions (notebooks/ → repo root)
# - in Google Colab, clones the latest codigo/ from GitHub into the session
#
# Repo (already filled for you):
#   https://github.com/Alberto-Rodriguez-Martinez/statistical-signal-processing (branch: main)

import os
import sys

# ---------- CONFIG: already filled -------------
GITHUB_OWNER = "Alberto-Rodriguez-Martinez"
GITHUB_REPO = "statistical-signal-processing"
GITHUB_BRANCH = "main"
# ----------------------------------------------

def in_colab():
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

# Ensure repo root is on sys.path so `import codigo` works when notebooks/ is the cwd.
# If your notebooks are directly in the repo root, this still works.
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# If running in Colab, fetch codigo/ from GitHub into the current working dir
if in_colab():
    # avoid re-cloning if codigo already present (e.g., user re-opened a saved notebook)
    if not os.path.exists("codigo"):
        print("Fetching codigo/ from GitHub for Colab session...")
        repo_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"
        tmp_dir = "tmp_repo_for_codigo"
        # shallow clone the branch
        os.system(f"git clone --depth 1 --branch {GITHUB_BRANCH} {repo_url} {tmp_dir} >/dev/null 2>&1 || true")
        if os.path.exists(tmp_dir):
            src = os.path.join(tmp_dir, "codigo")
            if os.path.exists(src):
                try:
                    os.rename(src, "codigo")
                except Exception:
                    # fallback: copy then remove
                    import shutil
                    shutil.copytree(src, "codigo")
            # cleanup
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print("Warning: could not clone the repository; check OWNER/REPO/BRANCH in the snippet.")
    # ensure the current dir is on sys.path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

# Now you can safely import codigo.*
try:
    import codigo   # noqa: F401
except Exception as e:
    print("Warning: could not import 'codigo'. If you are running locally, ensure the repo layout is correct.")
    print(e)

# If you want to explicitly print a short message for students:
print("Helper module 'codigo' is ready (or will be fetched in Colab)."),