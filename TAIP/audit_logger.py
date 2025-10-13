#!/usr/bin/env python3
"""
Audit Logger for the EGH455 TAIP Subsystem

This module provides database-backed audit logging functionality for the TAIP system.
It stores all system events with timestamps in a SQLite database and provides
search, filtering, and retrieval capabilities.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager
import threading

import config

# Database path
DB_PATH = config.TAIP_ROOT / "audit_logs.db"

class AuditLogger:
    """Thread-safe audit logging system with SQLite backend."""
    
    # Event types
    TELEMETRY = "telemetry"
    SYSTEM = "system"
    DRILL = "drill"
    CAMERA = "camera"
    SENSOR = "sensor"
    VISION = "vision"
    NETWORK = "network"
    ERROR = "error"
    
    # Event statuses
    INFO = "info"
    WARNING = "warning"
    ERROR_STATUS = "error"
    SUCCESS = "success"
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the audit logger.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses default.
        """
        self.db_path = db_path or DB_PATH
        self._lock = threading.Lock()
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON audit_logs(timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type 
                ON audit_logs(event_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON audit_logs(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON audit_logs(created_at DESC)
            """)
            
            conn.commit()
    
    def log(self, 
            event_type: str,
            action: str,
            details: str = "",
            status: str = INFO,
            metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (telemetry, system, drill, etc.)
            action: Short description of the action
            details: Detailed description
            status: Event status (info, warning, error, success)
            metadata: Additional metadata as a dictionary
            
        Returns:
            The ID of the inserted log entry
        """
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_logs 
                    (timestamp, event_type, action, details, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (timestamp, event_type, action, details, status, metadata_json))
                conn.commit()
                return cursor.lastrowid
    
    def get_logs(self,
                 limit: int = 100,
                 offset: int = 0,
                 event_type: Optional[str] = None,
                 status: Optional[str] = None,
                 search: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs with filtering and pagination.
        
        Args:
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            event_type: Filter by event type
            status: Filter by status
            search: Search in action and details fields
            start_date: Filter logs after this date (ISO format)
            end_date: Filter logs before this date (ISO format)
            
        Returns:
            List of log dictionaries
        """
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if search:
            query += " AND (action LIKE ? OR details LIKE ?)"
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern])
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                log_dict = dict(row)
                # Parse metadata JSON if present
                if log_dict.get('metadata'):
                    try:
                        log_dict['metadata'] = json.loads(log_dict['metadata'])
                    except json.JSONDecodeError:
                        log_dict['metadata'] = None
                logs.append(log_dict)
            
            return logs
    
    def get_log_count(self,
                      event_type: Optional[str] = None,
                      status: Optional[str] = None,
                      search: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> int:
        """
        Get the total count of logs matching the filters.
        
        Args:
            Same as get_logs()
            
        Returns:
            Total count of matching logs
        """
        query = "SELECT COUNT(*) as count FROM audit_logs WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if search:
            query += " AND (action LIKE ? OR details LIKE ?)"
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern])
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result['count'] if result else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about audit logs.
        
        Returns:
            Dictionary with various statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total events
            cursor.execute("SELECT COUNT(*) as count FROM audit_logs")
            total_events = cursor.fetchone()['count']
            
            # Events by type
            cursor.execute("""
                SELECT event_type, COUNT(*) as count 
                FROM audit_logs 
                GROUP BY event_type
            """)
            events_by_type = {row['event_type']: row['count'] for row in cursor.fetchall()}
            
            # Events by status
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM audit_logs 
                GROUP BY status
            """)
            events_by_status = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Recent events (last hour)
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM audit_logs 
                WHERE datetime(created_at) >= datetime('now', '-1 hour')
            """)
            events_last_hour = cursor.fetchone()['count']
            
            # Recent events (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM audit_logs 
                WHERE datetime(created_at) >= datetime('now', '-24 hours')
            """)
            events_last_day = cursor.fetchone()['count']
            
            return {
                'total_events': total_events,
                'events_by_type': events_by_type,
                'events_by_status': events_by_status,
                'events_last_hour': events_last_hour,
                'events_last_day': events_last_day
            }
    
    def clear_old_logs(self, days: int = 30) -> int:
        """
        Delete logs older than the specified number of days.
        
        Args:
            days: Number of days to keep logs
            
        Returns:
            Number of logs deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM audit_logs 
                    WHERE datetime(created_at) < datetime('now', ? || ' days')
                """, (f'-{days}',))
                conn.commit()
                return cursor.rowcount


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance (singleton)."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Convenience functions
def log_telemetry(action: str, details: str = "", status: str = AuditLogger.INFO, **metadata):
    """Log a telemetry event."""
    return get_audit_logger().log(AuditLogger.TELEMETRY, action, details, 
                                   status, metadata)

def log_system(action: str, details: str = "", status: str = AuditLogger.INFO, **metadata):
    """Log a system event."""
    return get_audit_logger().log(AuditLogger.SYSTEM, action, details, 
                                   status, metadata)

def log_drill(action: str, details: str = "", status: str = AuditLogger.INFO, **metadata):
    """Log a drill event."""
    return get_audit_logger().log(AuditLogger.DRILL, action, details, 
                                   status, metadata)

def log_vision(action: str, details: str = "", status: str = AuditLogger.INFO, **metadata):
    """Log a vision processing event."""
    return get_audit_logger().log(AuditLogger.VISION, action, details, 
                                   status, metadata)

def log_sensor(action: str, details: str = "", status: str = AuditLogger.INFO, **metadata):
    """Log a sensor event."""
    return get_audit_logger().log(AuditLogger.SENSOR, action, details, 
                                   status, metadata)

def log_network(action: str, details: str = "", status: str = AuditLogger.INFO, **metadata):
    """Log a network event."""
    return get_audit_logger().log(AuditLogger.NETWORK, action, details, 
                                   status, metadata)

def log_error(action: str, details: str = "", **metadata):
    """Log an error event."""
    return get_audit_logger().log(AuditLogger.ERROR, action, details, 
                                   AuditLogger.ERROR_STATUS, metadata)


# --- Standalone Test ---
if __name__ == '__main__':
    print("Testing Audit Logger...")
    
    # Create a test logger with a temporary database
    test_db = Path("/tmp/test_audit_logs.db")
    if test_db.exists():
        test_db.unlink()
    
    logger = AuditLogger(test_db)
    
    # Test logging various events
    print("\n1. Logging test events...")
    logger.log(AuditLogger.SYSTEM, "System Started", "TAIP system initialization", 
               AuditLogger.SUCCESS)
    logger.log(AuditLogger.TELEMETRY, "Pressure Reading", "Gauge pressure: 5.2 bar", 
               AuditLogger.INFO, {"pressure": 5.2})
    logger.log(AuditLogger.VISION, "Object Detection", "3 objects detected", 
               AuditLogger.INFO, {"count": 3})
    logger.log(AuditLogger.DRILL, "Drill Activated", "Pressure below threshold", 
               AuditLogger.WARNING, {"pressure": 1.5})
    logger.log(AuditLogger.ERROR, "Camera Error", "Failed to initialize OAK-D", 
               AuditLogger.ERROR_STATUS)
    
    # Test retrieval
    print("\n2. Retrieving all logs...")
    logs = logger.get_logs(limit=10)
    for log in logs:
        print(f"  [{log['timestamp']}] {log['event_type']}: {log['action']} - {log['details']}")
    
    # Test filtering
    print("\n3. Filtering by event type (vision)...")
    vision_logs = logger.get_logs(event_type=AuditLogger.VISION)
    print(f"  Found {len(vision_logs)} vision logs")
    
    # Test search
    print("\n4. Searching for 'pressure'...")
    search_results = logger.get_logs(search="pressure")
    print(f"  Found {len(search_results)} matching logs")
    
    # Test statistics
    print("\n5. Getting statistics...")
    stats = logger.get_stats()
    print(f"  Total events: {stats['total_events']}")
    print(f"  Events by type: {stats['events_by_type']}")
    print(f"  Events by status: {stats['events_by_status']}")
    
    # Test count
    print("\n6. Getting log count...")
    count = logger.get_log_count()
    print(f"  Total log count: {count}")
    
    print("\n✓ Audit logger test complete!")
    print(f"Test database: {test_db}")
