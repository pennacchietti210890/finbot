# run_gunicorn.py
import sys
import subprocess


def main():
    subprocess.run(["gunicorn", "ui.ui_dash:server", "--bind", "0.0.0.0:8502"])
