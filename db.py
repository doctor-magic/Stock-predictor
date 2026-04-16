import sqlite3
import json
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "scanner_cache.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            market_id TEXT,
            symbol TEXT,
            symbol_name TEXT,
            signal TEXT,
            confidence REAL,
            precision_score REAL,
            last_price REAL,
            scan_date DATE,
            PRIMARY KEY (market_id, symbol, scan_date)
        )
    """)
    conn.commit()
    conn.close()

def save_scan_results(market_id: str, results: list):
    """
    results is a list of dicts: 
    { 'symbol': ..., 'symbol_name': ..., 'signal': ..., 'confidence': ..., 'precision': ..., 'last_price': ... }
    """
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Optional: Delete old scans for this market to save space, or just keep history.
    # We will keep history but only query the latest.
    cursor.execute("DELETE FROM scan_results WHERE market_id = ? AND scan_date = ?", (market_id, today))
    
    for row in results:
        cursor.execute("""
            INSERT INTO scan_results (market_id, symbol, symbol_name, signal, confidence, precision_score, last_price, scan_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id, row['symbol'], row['symbol_name'], row['signal'], 
            row['confidence'], row['precision'], row['last_price'], today
        ))
    conn.commit()
    conn.close()

def get_latest_scan(market_id: str, min_confidence: float = 0.65, top_n: int = 10):
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM scan_results 
        WHERE market_id = ? AND scan_date = ? AND confidence >= ?
        ORDER BY confidence DESC
        LIMIT ?
    """, (market_id, today, min_confidence, top_n))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
