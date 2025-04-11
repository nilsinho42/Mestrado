"""
Script to run the API server.
"""

import sys
import time
from pathlib import Path

def run_servers():
    """Run the API server."""
    try:
        # Import and start the API server
        from api import start
        print("Starting API server...")
        start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_servers() 