# run_gunicorn.py
import subprocess
import sys


def main():
    subprocess.run(["gunicorn", "ui.ui_dash:server", "--bind", "0.0.0.0:8502"])
