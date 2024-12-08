import streamlit as st
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment
import base64
from io import BytesIO
import os
import yfinance as yf
import requests
from datetime import date
import numpy as np
import statistics
from scipy.stats import norm
import sys
import contextlib
from matplotlib.colors import PowerNorm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# https://pypi.org/project/QuantStats/
import quantstats as qs
qs.extend_pandas()

from hmmlearn import hmm
import matplotlib.gridspec as gridspec

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Momentum Index Strategy',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------

# Streamlit app
def main():
    st.title("Analysis for Stock Ticker")

    # User inputs
    ticker = st.text_input("Enter Ticker Symbol", "TECL")
    # SLOW = st.slider("Slow Rolling Average (in days)", 200, 300, 252)
    # FAST = st.slider("Fast Rolling Average (in days)", 20, 100, 21)
    #years_reviewed = st.slider("Years Reviewed", 5, 20, 8)
    SLOW = 252
    FAST = 21
    years_reviewed = 8

    # Step 1: Download data
    today_date = date.today()
    df = yf.download(ticker, start='1980-01-01', end=today_date)
    df['Change(%)'] = (df['Close'].diff(1) / df['Close'].shift(1)) * 100
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                       'Volume': 'volume', 'Adj Close': 'adj_close'}, inplace=True)
    df = df.sort_values('date')
    
    # Step 2: Data Processing
    df_adj = df.copy()
    df_adj['Rolling Change Sum - SLOW'] = df_adj['Change(%)'].rolling(window=SLOW).sum()
    df_adj['Rolling Change Sum - FAST'] = df_adj['Change(%)'].rolling(window=FAST).sum()
    df_adj = df_adj.dropna()

    # Step 3: Hidden Markov Model for market regimes
    dZ = df_adj[['date', 'Change(%)', 'close']]
    dZ['slow_rolling_avg'] = dZ['Change(%)'].rolling(window=SLOW).mean()
    dZ['fast_rolling_avg'] = dZ['Change(%)'].rolling(window=FAST).mean()
    dZ['price'] = (1 + dZ['Change(%)'] / 100).cumprod()
    dZ = dZ.dropna()

    # Apply HMM to predict market regimes
    X = dZ[['slow_rolling_avg', 'fast_rolling_avg']].values
    model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000)
    model.fit(X)
    hidden_states = model.predict(X)

    # Add hidden states to the dataframe
    dZ['hidden_state'] = hidden_states

    # Plot
    st.subheader("Market Regimes")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=dZ['date'], y='Change(%)', hue='hidden_state', data=dZ, palette='viridis', ax=ax)
    ax.set_title("Change(%) by Hidden State")
    ax.set_xlabel('Date')
    ax.set_ylabel('Change(%)')
    st.pyplot(fig)

    # Plot Rolling Averages
    st.subheader("Slow Rolling Average by Hidden State")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=dZ['date'], y='slow_rolling_avg', hue='hidden_state', data=dZ, palette='viridis', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Slow Rolling Average')
    st.pyplot(fig)
     # Plot Rolling Averages
    st.subheader("Fast Rolling Average by Hidden State")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=dZ['date'], y='fast_rolling_avg', hue='hidden_state', data=dZ, palette='viridis', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Fast Rolling Average')
    st.pyplot(fig)

    st.subheader("Rolling Averages by Hidden State")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=dZ['date'], y='price', hue='hidden_state', data=dZ, palette='viridis', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Average')
    st.pyplot(fig)

    st.subheader("Rolling Averages by Hidden State (Latest)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=dZ['date'].tail(100), y='price', hue='hidden_state', data=dZ.tail(100), palette='viridis', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Latest')
    st.pyplot(fig)

    

    # Show latest data in table
    # st.subheader(f"Latest 100 data points")
    # st.dataframe(dZ.tail(100))  # Displaying the last 100 rows

    # Save the chart as an image
    # fig.savefig('daily_chart.png')
    # st.download_button("Download Chart as PNG", data=open('daily_chart.png', 'rb'), file_name="daily_chart.png", mime="image/png")

if __name__ == "__main__":
    main()

