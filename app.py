from pathlib import Path
import runpy

# Forward execution to code/app.py
target = Path(__file__).resolve().parent / "code" / "app.py"
runpy.run_path(str(target), run_name="__main__")
