import os, pathlib

def ensure_setup():
    REPO_DIR = "/content/practical-nids"
    if pathlib.Path(REPO_DIR).exists():
        os.chdir(REPO_DIR)
