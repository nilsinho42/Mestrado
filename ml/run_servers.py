"""
Script to run the API server.
"""

import sys
import os
import uvicorn
import traceback
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run_servers():
    """Run the API server."""
    try:
        # Add the current directory to the system path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        # Import and start the API server
        print("Starting API server...")
        logger.debug("Starting uvicorn with api:app")
        
        # Run with increased timeouts for long-running requests
        uvicorn.run(
            "api:app", 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            timeout_keep_alive=120,  # Increase keep-alive timeout
            timeout_graceful_shutdown=30,  # Allow for graceful shutdown
            limit_max_requests=0,  # No limit on requests
            workers=1  # Single worker for simplicity
        )
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Detailed error:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_servers() 