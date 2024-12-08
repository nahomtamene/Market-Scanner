import sqlite3
import pandas as pd

def check_database():
    conn = sqlite3.connect('Equity.db')
    
    # List all unique tickers and their data counts
    print("\nTickers in database and their number of records:")
    query = """
    SELECT 
        Ticker,
        COUNT(*) as Days,
        MIN(AsOfDate) as StartDate,
        MAX(AsOfDate) as EndDate
    FROM EquityDailyPrice
    GROUP BY Ticker
    ORDER BY Ticker
    """
    tickers = pd.read_sql_query(query, conn)
    print(tickers)
    
    conn.close()

check_database() 