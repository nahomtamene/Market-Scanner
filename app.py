import dash
from dash import html, dcc, Input, Output, State, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import ta
import sqlite3
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Initialize the Dash app with callback exception suppression
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Database connection function
def get_stock_data(symbol=None, days=365):
    conn = sqlite3.connect('Equity.db')
    try:
        if symbol:
            query = """
            SELECT AsOfDate, Open, High, Low, Close, Volume
            FROM EquityDailyPrice
            WHERE Ticker = ?
            AND AsOfDate >= date('now', '-1 year')
            ORDER BY AsOfDate ASC
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            # Convert date and ensure it's in the correct format
            df['AsOfDate'] = pd.to_datetime(df['AsOfDate'])
            
            # Convert price columns to float
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert volume to integer
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').astype('Int64')
            
            if len(df) > 0:
                # Calculate technical indicators
                df['SMA20'] = df['Close'].rolling(window=20).mean()
                df['SMA50'] = df['Close'].rolling(window=50).mean()
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Hist'] = macd.macd_diff()
            
        else:
            df = pd.read_sql_query("SELECT DISTINCT Ticker FROM EquityDailyPrice", conn)
            
    except Exception as e:
        print(f"Error in get_stock_data: {str(e)}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    return df

# Technical Analysis Functions
def calculate_indicators(df, sma_period, rsi_period, macd_fast, macd_slow, macd_signal):
    if len(df) == 0:
        return df
    
    # Calculate SMA
    df['SMA'] = ta.trend.sma_indicator(df['Close'], window=sma_period)
    
    # Calculate RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
    
    # Calculate MACD
    df['MACD'] = ta.trend.macd_diff(df['Close'], 
                                   window_slow=macd_slow,
                                   window_fast=macd_fast,
                                   window_sign=macd_signal)
    return df

def check_conditions(df):
    """Check if the current price movement is significant"""
    if len(df) == 0:
        return False
    
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Get the latest return
    latest_return = df['Returns'].iloc[-1]
    
    # Consider it significant if the price moved more than 1%
    return abs(latest_return) > 0.01

# Define the layout
app.layout = html.Div([
    html.H1('Stock Scanner', style={'textAlign': 'center'}),
    
    html.Div([
        html.Button('Scan Market', id='scan-button', n_clicks=0,
                   style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px'})
    ], style={'margin': '20px', 'textAlign': 'center'}),
    
    # Add Criteria Form
    html.Div([
        html.H3('Filter Criteria'),
        dbc.Row([
            # Price Criteria
            dbc.Col([
                html.H5('Price Filters'),
                dbc.Input(id='min-price', placeholder='Min Price', type='number'),
                dbc.Input(id='max-price', placeholder='Max Price', type='number'),
            ], width=3),
            
            # Volume Criteria
            dbc.Col([
                html.H5('Volume Filters'),
                dbc.Input(id='min-volume', placeholder='Min Volume', type='number'),
                dbc.Input(id='max-volume', placeholder='Max Volume', type='number'),
            ], width=3),
            
            # Technical Indicators
            dbc.Col([
                html.H5('Technical Indicators'),
                dcc.Dropdown(
                    id='sma-condition',
                    options=[
                        {'label': 'Price > SMA20', 'value': 'above_sma20'},
                        {'label': 'Price < SMA20', 'value': 'below_sma20'},
                        {'label': 'Price > SMA50', 'value': 'above_sma50'},
                        {'label': 'Price < SMA50', 'value': 'below_sma50'},
                        {'label': 'SMA20 crosses above SMA50', 'value': 'golden_cross'},
                        {'label': 'SMA20 crosses below SMA50', 'value': 'death_cross'}
                    ],
                    multi=True,
                    placeholder='Select SMA Conditions'
                ),
                dcc.RangeSlider(
                    id='rsi-range',
                    min=0,
                    max=100,
                    step=1,
                    marks={0: '0', 30: '30', 70: '70', 100: '100'},
                    value=[30, 70],
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div(id='rsi-range-output'),
                dcc.Dropdown(
                    id='macd-condition',
                    options=[
                        {'label': 'MACD above Signal', 'value': 'macd_above'},
                        {'label': 'MACD below Signal', 'value': 'macd_below'},
                        {'label': 'MACD Crossover', 'value': 'macd_cross_above'},
                        {'label': 'MACD Crossunder', 'value': 'macd_cross_below'}
                    ],
                    multi=True,
                    placeholder='Select MACD Conditions'
                )
            ], width=6)
        ]),
        html.Button('Apply Filters', id='apply-filters', n_clicks=0,
                   style={'backgroundColor': '#2196F3', 'color': 'white', 
                          'padding': '10px 20px', 'margin': '20px'})
    ], style={'margin': '20px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    
    # Results table
    html.Div([
        dash_table.DataTable(
            id='stocks-table',
            columns=[
                {'name': 'Date', 'id': 'Date'},
                {'name': 'Ticker', 'id': 'Ticker'},
                {'name': 'Open', 'id': 'Open', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'High', 'id': 'High', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'Low', 'id': 'Low', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'Close', 'id': 'Close', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'Volume', 'id': 'Volume', 'type': 'numeric', 'format': {'specifier': '.0f'}}
            ],
            data=[],
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'cursor': 'pointer'
            },
            style_table={
                'height': '300px', 
                'overflowY': 'scroll',  # Enable vertical scrolling
                'overflowX': 'auto'    # Enable horizontal scrolling if needed
            },
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'minWidth': '100px',    # Minimum column width
                'width': '150px',       # Default column width
                'maxWidth': '180px',    # Maximum column width
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'position': 'sticky',   # Make headers sticky while scrolling
                'top': 0,
                'zIndex': 999
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Ticker'},
                    'textDecoration': 'underline',
                    'color': 'blue'
                }
            ],
            row_selectable=False,
            cell_selectable=True,
            page_size=500  # Show all rows, will scroll instead of paginate
        )
    ], id='table-container', style={'margin': '20px'}),
    
    # Stock chart with technical indicators
    dcc.Graph(id='stock-chart', style={'height': '800px'})
])

# Callback for scanning stocks and updating the table
@app.callback(
    Output('stocks-table', 'data'),
    [Input('scan-button', 'n_clicks'),
     Input('apply-filters', 'n_clicks')],
    [State('min-price', 'value'),
     State('max-price', 'value'),
     State('min-volume', 'value'),
     State('max-volume', 'value'),
     State('sma-condition', 'value'),
     State('rsi-range', 'value'),
     State('macd-condition', 'value')]
)
def scan_stocks(scan_clicks, filter_clicks, min_price, max_price, 
                min_volume, max_volume, sma_conditions, rsi_range, 
                macd_conditions):
    if scan_clicks == 0:
        return []
    
    conn = sqlite3.connect('Equity.db')
    try:
        # Build the query based on filters
        query = """
        WITH LatestPrices AS (
            SELECT 
                e.*,
                LAG(Close, 1) OVER (PARTITION BY Ticker ORDER BY AsOfDate) as prev_close,
                LAG(Close, 20) OVER (PARTITION BY Ticker ORDER BY AsOfDate) as sma20,
                LAG(Close, 50) OVER (PARTITION BY Ticker ORDER BY AsOfDate) as sma50
            FROM EquityDailyPrice e
            WHERE AsOfDate >= date('now', '-60 days')
        )
        SELECT DISTINCT
            Ticker, AsOfDate as Date, Open, High, Low, Close, Volume
        FROM LatestPrices
        WHERE 1=1
        """
        
        params = []
        if min_price is not None:
            query += " AND Close >= ?"
            params.append(min_price)
        if max_price is not None:
            query += " AND Close <= ?"
            params.append(max_price)
        if min_volume is not None:
            query += " AND Volume >= ?"
            params.append(min_volume)
        if max_volume is not None:
            query += " AND Volume <= ?"
            params.append(max_volume)
            
        query += " ORDER BY AsOfDate DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        
        # Apply technical indicator filters in Python
        if sma_conditions or rsi_range or macd_conditions:
            df = apply_technical_filters(df, sma_conditions, rsi_range, macd_conditions)
        
        results = df.to_dict('records')
        return results
        
    except Exception as e:
        print(f"Error in scan_stocks: {str(e)}")
        return []
    finally:
        conn.close()

def apply_technical_filters(df, sma_conditions, rsi_range, macd_conditions):
    # Calculate technical indicators
    if len(df) > 0:
        df['SMA20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).mean())
        df['SMA50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())
        df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.momentum.rsi(x, window=14))
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Apply filters
        if sma_conditions:
            for condition in sma_conditions:
                if condition == 'above_sma20':
                    df = df[df['Close'] > df['SMA20']]
                elif condition == 'below_sma20':
                    df = df[df['Close'] < df['SMA20']]
                # Add other SMA conditions...
                
        if rsi_range:
            df = df[(df['RSI'] >= rsi_range[0]) & (df['RSI'] <= rsi_range[1])]
            
        if macd_conditions:
            for condition in macd_conditions:
                if condition == 'macd_above':
                    df = df[df['MACD'] > df['MACD_Signal']]
                elif condition == 'macd_below':
                    df = df[df['MACD'] < df['MACD_Signal']]
                # Add other MACD conditions...
    
    return df

# Callback for displaying stock chart
@app.callback(
    Output('stock-chart', 'figure'),
    [Input('stocks-table', 'active_cell'),
     Input('stocks-table', 'data')]
)
def update_chart(active_cell, table_data):
    if not active_cell or not table_data or active_cell['column_id'] != 'Ticker':
        return {}
    
    row = table_data[active_cell['row']]
    ticker = row['Ticker']
    
    # Get one year of historical data
    df = get_stock_data(ticker, days=365)
    if len(df) == 0:
        return {}
    
    # Debug prints
    print("\nFirst 5 rows of data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.6, 0.2, 0.2])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['AsOfDate'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add SMAs
    fig.add_trace(
        go.Scatter(
            x=df['AsOfDate'],
            y=df['SMA20'],
            name='SMA20',
            line=dict(color='orange')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['AsOfDate'],
            y=df['SMA50'],
            name='SMA50',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=df['AsOfDate'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Add MACD
    fig.add_trace(
        go.Scatter(
            x=df['AsOfDate'],
            y=df['MACD'],
            name='MACD',
            line=dict(color='blue')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['AsOfDate'],
            y=df['MACD_Signal'],
            name='Signal',
            line=dict(color='orange')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df['AsOfDate'],
            y=df['MACD_Hist'],
            name='MACD Histogram',
            marker_color='gray'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - 1 Year Historical Data with Technical Indicators',
        yaxis_title='Price',
        yaxis2_title='RSI',
        yaxis3_title='MACD',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True) 