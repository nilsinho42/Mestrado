"""
Script to run the API server.
"""

import sys
import time
import os
import uvicorn
from pathlib import Path

def run_servers():
    """Run the API server."""
    try:
        # Add the current directory to the system path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        # Import and start the API server
        print("Starting API server...")
        # Run directly with uvicorn instead of using the api module's start function
        uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_servers() 