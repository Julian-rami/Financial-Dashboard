import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from Summary import select_stock
from datetime import datetime, timedelta

@st.cache_data
def list_wikipedia_sp500() -> pd.DataFrame:
    url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]

def select_stock_and_time():
    # Load the ticker list into session state if not already loaded
    if 'ticker_list' not in st.session_state:
        df_ticker = list_wikipedia_sp500()
        st.session_state['ticker_list'] = sorted(df_ticker.index.to_list())

    # Initialize session state for ticker selection and date range
    if 'ticker_select' not in st.session_state:
        st.session_state['ticker_select'] = st.session_state['ticker_list'][0]  # Default to the first ticker

    if 'timeinterval' not in st.session_state:
        st.session_state['timeinterval'] = 'Year'  # Default time interval

    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = datetime.today().date() - timedelta(days=365)

    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = datetime.today().date()

    # Sidebar: Ticker selection
    ticker_select = st.sidebar.selectbox(
        'Select Stock Ticker:',
        st.session_state['ticker_list'],
        index=st.session_state['ticker_list'].index(st.session_state['ticker_select'])
    )

    # Sidebar: Time interval selection
    timeinterval = st.sidebar.selectbox(
        'Time Interval',
        ['Year', 'Month', 'Day'],
        index=['Year', 'Month', 'Day'].index(st.session_state['timeinterval'])
    )

    # Determine days for time interval
    timeinterval_days = {'Year': 365, 'Month': 30, 'Day': 1}[timeinterval]

    # Sidebar: Date range inputs
    start_date = st.sidebar.date_input(
        "Start date",
        value=st.session_state['start_date']
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=st.session_state['end_date']
    )

    # Update session state and fetch new data if button is pressed or inputs change
    if (ticker_select != st.session_state['ticker_select'] or
        timeinterval != st.session_state['timeinterval'] or
        start_date != st.session_state['start_date'] or
        end_date != st.session_state['end_date'] or
        st.sidebar.button('Update Data')):
        
        st.session_state['ticker_select'] = ticker_select
        st.session_state['timeinterval'] = timeinterval
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date

        # Fetch stock data
        st.session_state['stock_data'] = yf.Ticker(ticker_select).history(start=start_date, end=end_date)

    # Return the selected stock data and time interval
    return st.session_state['stock_data'], st.session_state['timeinterval']

stock, timeinterval = select_stock_and_time()

st.subheader('Charts')

fig = make_subplots(specs=[[{"secondary_y":True}]])

line_plot = go.Scatter(x=stock.index, y=stock["Close"], name='Closing Price')

candle_plot = go.Candlestick(x = stock.index,
                             open = stock["Open"],
                             high = stock["High"],
                             low = stock["Low"],
                             close = stock["Close"],
                             name= 'Candlestick')

bar_plot = go.Bar(x=stock.index, y=stock["Volume"], name='Volume',marker_color=np.where(stock['Close'].pct_change() > 0, 'green', 'red'))

ma_plot = go.Scatter(x=stock.index, y=stock['Close'].rolling(window=50).mean(), name='50-Day Moving Average')

fig.add_trace(line_plot, secondary_y=True)
fig.add_trace(candle_plot, secondary_y=True)
fig.add_trace(ma_plot, secondary_y=True)
fig.add_trace(bar_plot, secondary_y=False)

fig.update_traces(visible=False)
fig.update_traces(visible=True, selector=dict(name="Line Plot"))
fig.update_traces(visible=False, selector=dict(name="Candle Stick Plot"))

my_buttons = [{'label': 'Line Plot', 'method': 'update', 'args': [{'visible': [True, False, True, True]}]},
              {'label': 'Candle Stick Plot', 'method': 'update', 'args': [{'visible': [False, True, True, True]}]}]

def tick():
    if timeinterval == 'Day':
        return 'D7'
    elif timeinterval == 'Month':
        return 'M1'
    elif timeinterval == 'Year':
        return 'M6'

fig.update_layout({'updatemenus': [{'type': 'buttons', 
                                    'direction': 'down', 
                                    'x': 1.2, 'y': 0.5, 
                                    'showactive': True, 
                                    'active': 0, 
                                    'buttons': my_buttons}],
                        'xaxis':   {
                                    'rangeslider': {'visible': False}, 
                                    'rangebreaks': [],                  
                                    'tickmode': 'linear',
                                    'dtick': tick(), 
                                    'tickformat': '%b %Y'             
                                    }})

max_volume = stock['Volume'].max()
fig.update_yaxes(range=[0, max_volume * 1.6], showticklabels=False,secondary_y=False)

date_buttons = [{'count': 1, 'step': 'month', 'stepmode': 'backward', 'label': '1M'}, {'count': 3, 'step': 'month', 'stepmode': 'backward', 'label': '3M'},
                {'count': 6, 'step': 'month', 'stepmode': 'backward', 'label': '6M'}, {'count': 1, 'step': 'year', 'stepmode': 'backward', 'label': '1Y'},
                {'count': 3, 'step': 'year', 'stepmode': 'backward', 'label': '3Y'}, {'count': 5, 'step': 'year', 'stepmode': 'backward', 'label': '5Y'},
                {'step': 'all', 'label': 'MAX'}]

fig.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})

fig.update_layout(width= 1000, height= 600)

st.plotly_chart(fig)