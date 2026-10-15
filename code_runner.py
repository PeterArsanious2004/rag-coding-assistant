import os
import subprocess
import sys
import tempfile


def run_python(code: str, timeout: int = 10) -> dict:
    """Runs code in a separate process and captures stdout and stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True, text=True, timeout=timeout, cwd=tmp,
            )
            return {"stdout": p.stdout, "stderr": p.stderr, "returncode": p.returncode}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1}