import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
from pages.Chart import select_stock_and_time
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf, acf
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.subheader('Additional Statistics')

# Load the stock data using your existing select_stock function
stock_data, _ = select_stock_and_time()

def calculate_rsi(data, window=14):
    # Calculate daily price changes
    delta = data['Close'].diff()
    
    # Separate positive and negative gains
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Calculate the Relative Strength (RS) and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Calculate RSI for stock data
stock_data['RSI'] = calculate_rsi(stock_data)

# Create the RSI plot
def plot_rsi(data):
    fig = go.Figure()

    # Add RSI line
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    
    # Add overbought and oversold thresholds
    fig.add_shape(type="line", x0=data.index[0], y0=70, x1=data.index[-1], y1=70, line=dict(color="red", dash="dash"), name="Overbought")
    fig.add_shape(type="line", x0=data.index[0], y0=30, x1=data.index[-1], y1=30, line=dict(color="green", dash="dash"), name="Oversold")
    
    # Update layout
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        yaxis_title="RSI Value",
        xaxis_title="Date",
        template="plotly_white",
    )

    return fig

# Display RSI plot in Streamlit
st.plotly_chart(plot_rsi(stock_data))

# Info button with an expander
with st.expander("What is RSI?"):
    st.write("""
        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. 
        It oscillates between 0 and 100. Typically:
        
        - An RSI above 70 suggests that the asset is becoming overbought or overvalued.
        - An RSI below 30 indicates that the asset may be oversold or undervalued.
        
        Traders often use these levels to identify potential reversals or trend changes.
    """)

returns = stock_data["Close"].pct_change().dropna()

# KDE for PDF
kde = gaussian_kde(returns)
x_vals = np.linspace(returns.min(), returns.max(), 500)
pdf = kde(x_vals)

# Plot PDF with Histogram
fig = go.Figure()

# Histogram of Returns
fig.add_trace(go.Histogram(
    x=returns,
    histnorm='probability density',
    name="Histogram",
    opacity=0.6
))

# KDE Curve
fig.add_trace(go.Scatter(
    x=x_vals,
    y=pdf,
    mode='lines',
    name="KDE Curve",
    line=dict(color='firebrick', width=2)
))

# Add Statistical Markers
mean_return = returns.mean()
fig.add_vline(x=mean_return, line=dict(color="blue", dash="dash"), annotation_text="Mean")

# Customize Layout
fig.update_layout(
    title=f"Daily Returns PDF",
    xaxis_title="Returns",
    yaxis_title="Density",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig)

# Add PDF Explanation in an Expander
with st.expander("What is a Probability Density Function (PDF)?"):
    st.markdown("""
    **A Probability Density Function (PDF)** provides a way to understand the likelihood of a continuous random variable 
    (like stock returns) taking on a particular range of values. Here's what it means:
    
    - **Histogram vs PDF**: The histogram shows observed frequencies of returns, while the PDF (represented by the KDE curve) is a smooth approximation.
    - **Key Insight**: The area under the PDF curve between two points gives the probability that the random variable falls within that range.
    - **Applications in Finance**: Understanding the PDF of stock returns can help assess risk, calculate probabilities of extreme losses, and model future behavior.
    
    For example, if the PDF suggests most returns cluster around 0.02 (2%), the stock might have a modest growth trend. Conversely, a wide spread in the PDF indicates higher volatility.
    """)

# Display Statistics
with st.expander("Daily Returns Summary"):
    st.write(f"Mean: {mean_return:.2%}")
    st.write(f"Standard Deviation: {returns.std():.2%}")
    st.write(f"Skewness: {returns.skew():.2f}")
    st.write(f"Kurtosis: {returns.kurtosis():.2f}")

arima_data = stock_data['Close'].diff().dropna()

# ARIMA Modeling
st.sidebar.subheader("ARIMA Configuration")
st.sidebar.markdown("Use the ACF (for p) and PACF (for q) plots below to determine suitable ARIMA parameters.")
p = st.sidebar.slider("AR Order (p)", min_value=0, max_value=24, value=0)
d = st.sidebar.slider("Differencing Order (d)", min_value=0, max_value=1, value=0)
q = st.sidebar.slider("MA Order (q)", min_value=0, max_value=24, value=0)
forecast_steps = st.sidebar.slider("Forecast Steps", min_value=5, max_value=365, value=30)

# Compute ACF and PACF
acf_values = acf(arima_data, nlags=24, fft=True)
pacf_values = pacf(arima_data, nlags=24)
lags = np.arange(len(acf_values))

def acf_pacf():
    # Calculate the significance bands
    N = len(arima_data)
    conf_level = 1.96 / np.sqrt(N)

    # Create ACF and PACF plots with significance bands
    acf_pacf_fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

    # Add ACF plot with significance bands
    acf_pacf_fig.add_trace(
        go.Bar(x=lags, y=acf_values, name="ACF", marker_color="blue"),
        row=1, col=1
    )
    acf_pacf_fig.add_trace(
        go.Scatter(
            x=lags, y=[conf_level] * len(lags), mode="lines", name="+95% Conf", line=dict(color="green", dash="dash")
        ),
        row=1, col=1
    )
    acf_pacf_fig.add_trace(
        go.Scatter(
            x=lags, y=[-conf_level] * len(lags), mode="lines", name="-95% Conf", line=dict(color="green", dash="dash")
        ),
        row=1, col=1
    )

    # Add PACF plot with significance bands
    acf_pacf_fig.add_trace(
        go.Bar(x=lags, y=pacf_values, name="PACF", marker_color="orange"),
        row=1, col=2
    )
    acf_pacf_fig.add_trace(
        go.Scatter(
            x=lags, y=[conf_level] * len(lags), mode="lines", name="+95% Conf", line=dict(color="green", dash="dash"),
            showlegend=False
        ),
        row=1, col=2
    )
    acf_pacf_fig.add_trace(
        go.Scatter(
            x=lags, y=[-conf_level] * len(lags), mode="lines", name="-95% Conf", line=dict(color="green", dash="dash"),
            showlegend=False
        ),
        row=1, col=2
    )

    # Update layout for ACF and PACF plots
    acf_pacf_fig.update_layout(
        title_text="ACF and PACF Plots with Significance Bands",
        template="plotly_white",
        xaxis_title="Lag",
        yaxis_title="Correlation",
        xaxis2_title="Lag",
        yaxis2_title="Partial Correlation"
    )

    # Streamlit integration
    return st.plotly_chart(acf_pacf_fig)

acf_pacf()

# ARIMA Explanation
with st.expander("What is ARIMA?"):
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)** is a statistical method used for time series forecasting.
    
    - **Components**:
      - **AR (p)**: The number of lag observations included in the model.
      - **I (d)**: The number of times the data needs to be differenced to make it stationary.
      - **MA (q)**: The number of lagged forecast errors in the prediction model.
    
    - **How It Works**:
      ARIMA models patterns in the historical data and projects them into the future. It is suitable for univariate time series data that exhibits a clear trend or seasonality.
      Stationarity is a key requirement of ARIMA forecasting therefore the Stock Prices are differentiated to meet this requirement.
      It is not the actual price that is forecasted but the price changes or the price delta.  
    """)

def arima(arima_data, p, d, q):
    # Train/Test Split
    train_size = int(len(arima_data) * 0.8)
    train, test = arima_data[:train_size], arima_data[train_size:]

    try:
        # Fit ARIMA Model
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()

        # Test Data Fit
        test_forecast_obj = model_fit.get_forecast(steps=len(test))
        test_forecast = test_forecast_obj.predicted_mean
        test_conf_int = test_forecast_obj.conf_int()
        test_forecast_index = test.index
        test_forecast_series = pd.Series(test_forecast.values, index=test_forecast_index)

        forecast_obj = model_fit.get_forecast(steps=forecast_steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        forecast_index = pd.date_range(start=test_forecast_index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="B")

        # Combine Forecast with Actual Data
        forecast_series = pd.Series(data=forecast.values, index=forecast_index, name="Forecast")

        # Combine actual test data, test forecast, and extended forecast
        combined_forecast = pd.concat([test_forecast_series, forecast_series])
        combined_actual_and_forecast = pd.concat([test, combined_forecast], axis=1)
        combined_actual_and_forecast.columns = ["Actual", "Forecast"]

        # Confidence Intervals for Forecast
        lower_series = pd.Series(conf_int.iloc[:, 0].values, index=forecast_index, name="Lower Bound")
        upper_series = pd.Series(conf_int.iloc[:, 1].values, index=forecast_index, name="Upper Bound")

        mse = mean_squared_error(test, test_forecast_series)
        mae = mean_absolute_error(test, test_forecast_series)

        # Plot Results
        fig = go.Figure()

        # Plot Actual Data
        fig.add_trace(go.Scatter(x=arima_data.index, y=arima_data, mode='lines', name='Price Delta', line=dict(color='blue')))

        # Plot Forecast for Test Set
        fig.add_trace(go.Scatter(
        x=test_forecast_series.index, y=test_forecast_series, 
        mode='lines', name='Forecast (Test Set)', 
        line=dict(color='red', dash='dash')))

        # Plot Forecast
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_series, mode='lines', name='Forecast', line=dict(color='green', dash='dash')))

        # Add Confidence Interval as Shaded Area
        fig.add_trace(go.Scatter(
            x=np.concatenate([forecast_index, forecast_index[::-1]]),
            y=np.concatenate([upper_series, lower_series[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 128, 0, 0.2)',  # Green shaded area
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

        # Customize Layout
        fig.update_layout(
            title="ARIMA Forecast with Test Set Comparison",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return model_fit, mse, mae, fig
    
    except Exception as e:
        st.error(f"An error occurred while fitting the ARIMA model: {e}")

        return None, None, None, None
        
model_fit, mse, mae, fig = arima(arima_data, p, d, q)

if model_fit is not None:
    # Display Plot
    st.plotly_chart(fig)
    # Display Model Summary
    with st.expander("ARIMA Model Summary"):
        st.text(model_fit.summary())
        st.write("### Model Performance Metrics")
        st.write(f"- **Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"- **Mean Absolute Error (MAE):** {mae:.2f}")

else:
    st.error("ARIMA model fitting failed. Please adjust the parameters.") 

    
