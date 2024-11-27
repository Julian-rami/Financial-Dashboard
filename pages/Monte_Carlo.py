import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
from Summary import select_stock

st.markdown(
    f"""
    <div style='text-align: center; background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd; border-radius: 10px;'>
        <h3 style='color: #000; margin-bottom: 5px;'>Selected Ticker:</h3>
        <h2 style='color: #FF5733; margin-top: 0;'>{st.session_state["ticker_select"]}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

def run_simulation(stock_price, time_horizon, n_simulation):

    seed = np.random.seed(123)

    daily_vola = stock_price['Close'].pct_change().std()

    # Initialize array to store simulations (n_simulation x time_horizon)
    simulations = np.zeros((time_horizon, n_simulation))
    #df_simulation = pd.DataFrame()

    last_price = stock_price['Close'].iloc[-1]

    for i in range(n_simulation):
        d_return = np.random.normal(loc=0, scale=daily_vola, size=time_horizon)

        price_path = last_price * np.cumprod(1 + d_return)

        simulations[:, i] = price_path
        #df_simulation[i] = price_path

    df_simulation = pd.DataFrame(
        simulations, 
        index=pd.date_range(start=stock_price.index[-1], periods=time_horizon, freq='D')
    )

    return df_simulation 

def plot_simulation_price(stock_price, df_simulation):

    fig = go.Figure()

    for i in df_simulation.columns:
        fig.add_trace(go.Scatter(x=df_simulation.index, y=df_simulation[i], showlegend=False, hoverinfo='skip'))

    current_price = stock_price['Close'].iloc[-1]

    fig.add_shape(
        type="line",
        x0=df_simulation.index[0], x1=df_simulation.index[-1],  # Line spans entire x-axis
        y0=current_price, y1=current_price,
        line=dict(color="blue"),
        name='Current Price'
    )

    fig.update_layout(
        title="Monte Carlo Simulated Stock Price Paths",
        xaxis_title="Days",
        yaxis_title="Price",
        template="plotly_white"
    )

    return st.plotly_chart(fig)

def var(df_simulation, stock_price):
    ending_price = df_simulation.iloc[-1:, :].values[0, ] #values converts the df into a numpy array

    # Price at 95% confidence interval
    future_price_95ci = np.percentile(ending_price, 5)

    # Value at Risk
    # 95% of the time, the losses will not be more than 16.35 USD
    VaR = stock_price['Close'].iloc[-1] - future_price_95ci

    return st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

stock_price = select_stock()

# Sidebar controls for simulations
st.sidebar.subheader('Monte Carlo Simulation Controls')

# Number of simulations selection
n_simulation = st.sidebar.selectbox("Select Number of Simulations", [200, 500, 1000], index=2)

# Time range selection
time_horizon = st.sidebar.selectbox("Select Time Range (in Days)", [30 , 60 , 90], index=2)

df_simulation = run_simulation(stock_price, time_horizon, n_simulation)
plot_simulation_price(stock_price, df_simulation)
var(df_simulation, stock_price)
