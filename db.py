import sqlite3
import os
from datetime import date

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

def get_latest_scan(market_id: str):
    """Return all cached results for market_id from today, or empty list."""
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM scan_results 
        WHERE market_id = ? AND scan_date = ?
        ORDER BY confidence DESC
    """, (market_id, today))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "symbol": r["symbol"],
        "symbol_name": r["symbol_name"],
        "signal": r["signal"],
        "confidence": r["confidence"],
        "precision": r["precision_score"],
        "last_price": r["last_price"]
    } for r in rows]

