import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def get_sp500_tickers():
    # Get S&P 500 tickers from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    df = table[0]
    return df['Symbol'].tolist()

def insert_sample_data():
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} S&P 500 stocks")
    
    conn = sqlite3.connect('Equity.db')
    cursor = conn.cursor()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Downloading data for {ticker}...")
            # Get data from yfinance
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if stock.empty:
                print(f"No data found for {ticker}, skipping...")
                continue
            
            # Insert data
            for index, row in stock.iterrows():
                try:
                    cursor.execute('''
                    INSERT OR REPLACE INTO EquityDailyPrice 
                    (Ticker, AsOfDate, Open, High, Low, Close, Volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        index.strftime('%Y-%m-%d'),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume'])
                    ))
                except Exception as e:
                    print(f"Error inserting {ticker} data for {index}: {str(e)}")
                    continue
            
            conn.commit()
            print(f"Successfully inserted data for {ticker}")
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    conn.close()
    print("Data insertion complete!")

if __name__ == "__main__":
    insert_sample_data() 