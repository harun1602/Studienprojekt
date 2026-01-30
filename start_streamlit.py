from pathlib import Path
import subprocess

def detect_project_dir() -> Path:
    cwd = Path.cwd()

    # Fall 1: Du bist bereits im Projektordner
    if (cwd / "streamlit_app.py").exists() and (cwd / "venv" / "Scripts" / "python.exe").exists():
        return cwd

    # Fall 2: Du bist im Elternordner und der Projektordner liegt darunter
    candidate = cwd / "DemonstratorProzesszeitprognose"
    if (candidate / "streamlit_app.py").exists() and (candidate / "venv" / "Scripts" / "python.exe").exists():
        return candidate

    raise FileNotFoundError(
        "Projektordner nicht gefunden. Starte das Skript entweder:\n"
        " - im Projektordner (wo streamlit_app.py liegt)\n"
        " - oder im Ordner darüber (wo der Ordner DemonstratorProzesszeitprognose liegt)."
    )

def main():
    project_dir = detect_project_dir()
    venv_python = project_dir / "venv" / "Scripts" / "python.exe"

    cmd = [str(venv_python), "-m", "streamlit", "run", "streamlit_app.py"]
    subprocess.run(cmd, cwd=str(project_dir), check=True)

if __name__ == "__main__":
    main()
