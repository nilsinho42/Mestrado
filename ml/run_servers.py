import subprocess
import sys
import time
from pathlib import Path

def run_servers():
    # Get the current directory
    current_dir = Path(__file__).parent

    try:
        # Start FastAPI server
        fastapi_process = subprocess.Popen(
            [sys.executable, str(current_dir / "api" / "main.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("FastAPI server started on http://localhost:8000")

        # Start Streamlit server
        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", 
             str(current_dir / "cloud_comparison" / "streamlit_app.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Streamlit server started on http://localhost:8501")

        # Keep the script running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down servers...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    run_servers() 