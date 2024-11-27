import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "majorHoldersBreakdown,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown,"
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret


st.title('Stock Dashboard')

@st.cache_data
def list_wikipedia_sp500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]

# Initialize ticker list in session state
if 'ticker_list' not in st.session_state:
    df_ticker = list_wikipedia_sp500()
    st.session_state.ticker_list = sorted(df_ticker.index.to_list())

def select_stock():
    # Load ticker data and sort the list
    df_ticker = list_wikipedia_sp500()
    ticker_list_sorted = sorted(df_ticker.index.to_list())

    # Sidebar select box to choose the ticker
    ticker_select = st.sidebar.selectbox('Select Stock Ticker:', ticker_list_sorted)

    # Update button to fetch new data
    if st.sidebar.button('Update Data'):
        stock = yf.Ticker(ticker_select).history(start='1900-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        # Store the updated stock data in Streamlit's session state
        st.session_state['stock_data'] = stock
        st.session_state['ticker_select'] = ticker_select
        
    # Default to loading the existing ticker if already in session state
    if 'stock_data' not in st.session_state:
        stock = yf.Ticker(ticker_select).history(start='1900-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        st.session_state['stock_data'] = stock
        st.session_state['ticker_select'] = ticker_select

    return st.session_state['stock_data']

stock = select_stock()

st.subheader('Summary')

fig = make_subplots(specs=[[{"secondary_y":True}]])
area_plot = go.Scatter(x = stock.index, y = stock["Close"], name= 'Closing Price',
                       fill='tozeroy', fillcolor='rgba(133, 133, 241, 0.2)', showlegend=False)
fig.add_trace(area_plot, secondary_y=True)

bar_plot = go.Bar(x = stock.index, y = stock["Volume"], name='Volume',marker_color=np.where(stock['Close'].pct_change() > 0, 'green', 'red'), showlegend=False)

fig.add_trace(bar_plot, secondary_y=False)

max_volume = stock['Volume'].max()
fig.update_yaxes(range=[0, max_volume * 1.6], showticklabels=False,secondary_y=False)

date_buttons = [{'count': 1, 'step': 'month', 'stepmode': 'todate', 'label': '1M'}, {'count': 3, 'step': 'month', 'stepmode': 'todate', 'label': '3M'},
                {'count': 6, 'step': 'month', 'stepmode': 'todate', 'label': '6M'}, {'count': 1, 'step': 'year', 'stepmode': 'todate', 'label': '1Y'},
                {'count': 3, 'step': 'year', 'stepmode': 'todate', 'label': '3Y'}, {'count': 5, 'step': 'year', 'stepmode': 'todate', 'label': '5Y'},
                {'step': 'all', 'label': 'MAX'}]

fig.update_layout({'xaxis': {
                    'rangeselector': {'buttons': date_buttons},
                    'range': [stock.index.min(), stock.index.max()],}
                    })

st.plotly_chart(fig)

summary_dict = yf.Ticker(st.session_state.ticker_select).info

summary_data1 = {
    'Previous Close': summary_dict.get('previousClose'),
    'Open': summary_dict.get('open'),
    'Bid': summary_dict.get('bid'),
    'Ask': summary_dict.get('ask')}
summary_data2 = {'Day Low': summary_dict.get('dayLow'),
    'Day High': summary_dict.get('dayHigh'),
    '52 Week Low': summary_dict.get('fiftyTwoWeekLow'),
    '52 Week High': summary_dict.get('fiftyTwoWeekHigh')}
summary_data3 = {'Volume': summary_dict.get('volume'),
    'Avg Volume': summary_dict.get('averageVolume'),
    'Market Cap': summary_dict.get('marketCap'),
    'Beta': summary_dict.get('beta')}
summary_data4= {'Trailing Peg': summary_dict.get('trailingPegRatio'),
    'PE-Ratio': summary_dict.get('pegratio'),
    'Trailing EPS': summary_dict.get('trailingEps'),
    'Forward EPS': summary_dict.get('forwardEps')}
summary_data5= {'Dividend Rate': summary_dict.get('dividendRate'),
    'Dividend Yield': summary_dict.get('dividendYield'),
    'Ex-Dividend Date': summary_dict.get('exDividendDate'),
    'Last Split Factor': summary_dict.get('lastSplitFactor')
}

col1, col2, col3, col4, col5 = st.columns([4, 4, 4, 4, 4])


summary_df1 = pd.DataFrame.from_dict(summary_data1, orient='index').round(2)
summary_df1.columns = [""]
summary_df2 = pd.DataFrame.from_dict(summary_data2, orient='index').round(2)
summary_df2.columns = [""]
summary_df3 = pd.DataFrame.from_dict(summary_data3, orient='index').round(2)
summary_df3.columns = [""]
summary_df4 = pd.DataFrame.from_dict(summary_data4, orient='index').round(2)
summary_df4.columns = [""]
summary_df5 = pd.DataFrame.from_dict(summary_data5, orient='index').round(2)
summary_df5.columns = [""]

st.subheader('Stock Summary Information')

with col1:
    st.table(summary_df1)

with col2:
    st.table(summary_df2)

with col3:
    st.table(summary_df3)

with col4:
    st.table(summary_df4)

with col5:
    st.table(summary_df5)

st.subheader('Company Description')
info_data = {'longBusinessSummary': summary_dict.get('longBusinessSummary')}
info_value = info_data['longBusinessSummary']
st.write(info_value)