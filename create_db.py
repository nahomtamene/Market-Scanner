import sqlite3

def create_database():
    conn = sqlite3.connect('Equity.db')
    cursor = conn.cursor()
    
    # Drop the existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS EquityDailyPrice')
    
    # Create the table with the correct schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS EquityDailyPrice (
        Ticker TEXT,
        AsOfDate DATE,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Volume INTEGER,
        PRIMARY KEY (Ticker, AsOfDate)
    )
    ''')
    
    conn.commit()
    conn.close()

create_database() 