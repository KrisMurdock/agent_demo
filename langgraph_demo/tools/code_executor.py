import subprocess
import sys
import tempfile
import os

BLOCKED_PATTERNS = ["os.system", "subprocess", "shutil.rmtree", "os.remove", "os.unlink"]


def execute_python_code(code: str, timeout: int = 5) -> dict:
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return {
                "stdout": "",
                "stderr": f"Blocked: code contains forbidden pattern '{pattern}'",
                "success": False,
                "timeout": False,
            }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "success": False,
            "timeout": True,
        }
    finally:
        os.unlink(tmp_path)
