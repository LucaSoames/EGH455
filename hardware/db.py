#!/usr/bin/env python3
import sqlite3
from sqlite3 import Connection
from typing import Optional, Dict

# Path to the SQLite database file
DB_PATH = "sensor_data.db"


def get_connection() -> Connection:
    """Create a SQLite connection and enable row access by column name."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database, creating the readings table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL,
            humidity REAL,
            gas_reducing REAL,
            gas_oxidising REAL,
            gas_nh3 REAL,
            light REAL,
            proximity INTEGER
        );
        """
    )
    conn.commit()
    conn.close()


def insert_reading(
    timestamp: str,
    temperature: float,
    humidity: float,
    gas_reducing: float,
    gas_oxidising: float,
    gas_nh3: float,
    light: float,
    proximity: int
) -> None:
    """Insert a new sensor reading into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO readings
          (timestamp, temperature, humidity,
           gas_reducing, gas_oxidising, gas_nh3,
           light, proximity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, temperature, humidity,
         gas_reducing, gas_oxidising, gas_nh3,
         light, proximity)
    )
    conn.commit()
    conn.close()


def get_latest_reading() -> Optional[Dict]:
    """Fetch the most recent sensor reading from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM readings ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
