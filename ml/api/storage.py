import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import sqlite3
import logging

logger = logging.getLogger(__name__)

class ProcessingStorage:
    def __init__(self, db_path: str = "processing_status.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create processing status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_status (
                processing_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                video_path TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error TEXT,
                results TEXT
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status_user 
            ON processing_status(status, user_id)
        """)
        
        conn.commit()
        conn.close()

    def create_status(self, processing_id: str, video_path: str, user_id: int) -> None:
        """Create initial processing status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO processing_status 
                (processing_id, status, video_path, user_id)
                VALUES (?, ?, ?, ?)
            """, (processing_id, "pending", video_path, user_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error creating status: {str(e)}")
            raise
        finally:
            conn.close()

    def update_status(self, processing_id: str, status: str, error: Optional[str] = None, results: Optional[Dict[str, Any]] = None) -> None:
        """Update processing status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            results_json = json.dumps(results) if results else None
            cursor.execute("""
                UPDATE processing_status 
                SET status = ?, error = ?, results = ?, updated_at = CURRENT_TIMESTAMP
                WHERE processing_id = ?
            """, (status, error, results_json, processing_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating status: {str(e)}")
            raise
        finally:
            conn.close()

    def get_status(self, processing_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT status, error, results, created_at, updated_at
                FROM processing_status
                WHERE processing_id = ?
            """, (processing_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            status, error, results_json, created_at, updated_at = row
            results = json.loads(results_json) if results_json else None
            
            return {
                "status": status,
                "error": error,
                "results": results,
                "created_at": created_at,
                "updated_at": updated_at
            }
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            raise
        finally:
            conn.close()

    def get_user_statuses(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing statuses for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT processing_id, status, error, results, created_at, updated_at
                FROM processing_status
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            statuses = []
            
            for row in rows:
                processing_id, status, error, results_json, created_at, updated_at = row
                results = json.loads(results_json) if results_json else None
                
                statuses.append({
                    "processing_id": processing_id,
                    "status": status,
                    "error": error,
                    "results": results,
                    "created_at": created_at,
                    "updated_at": updated_at
                })
            
            return statuses
        except Exception as e:
            logger.error(f"Error getting user statuses: {str(e)}")
            raise
        finally:
            conn.close()

    def cleanup_old_statuses(self, days: int = 30) -> None:
        """Remove processing statuses older than specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM processing_status
                WHERE updated_at < datetime('now', ?)
            """, (f'-{days} days',))
            conn.commit()
        except Exception as e:
            logger.error(f"Error cleaning up old statuses: {str(e)}")
            raise
        finally:
            conn.close() 