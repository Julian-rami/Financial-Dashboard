import streamlit as st
import yfinance as yf
from Summary import select_stock



st.subheader('Financial Data')

stock = select_stock()

# Create two columns for placing the financial statement and time frequency options in the same row
col1, col2 = st.columns([3, 2])

# Financial statement selection (column 1)
with col1:
    view_option = st.radio(
        "Select Financial Statement:",
        ('Income Statement', 'Balance Sheet', 'Cash Flow')
    )

# Time frequency selection (column 2) with horizontal layout
with col2:
    time_option = st.radio(
        "Time Frequency:",
        ('Annual', 'Quarterly'),
        horizontal=True  # Make buttons horizontal
    )

# Determine the appropriate financial data and whether it's quarterly or annual
if view_option == 'Income Statement':
    financial_data = (yf.Ticker(st.session_state.ticker_select).financials 
                      if time_option == 'Annual' 
                      else yf.Ticker(st.session_state.ticker_select).quarterly_financials)
    is_quarterly = (time_option == 'Quarterly')

elif view_option == 'Balance Sheet':
    financial_data = (yf.Ticker(st.session_state.ticker_select).balance_sheet 
                      if time_option == 'Annual' 
                      else yf.Ticker(st.session_state.ticker_select).quarterly_balance_sheet)
    is_quarterly = (time_option == 'Quarterly')

elif view_option == 'Cash Flow':
    financial_data = (yf.Ticker(st.session_state.ticker_select).cashflow 
                      if time_option == 'Annual' 
                      else yf.Ticker(st.session_state.ticker_select).quarterly_cashflow)
    is_quarterly = (time_option == 'Quarterly')

# Display the financial data
st.write(financial_data)
