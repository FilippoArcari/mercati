import pandas as pd
import yfinance as yf

# let's download just aapl for 10 days
raw = yf.download(["AAPL"], start="2010-01-01", end="2010-01-10", interval="1d", group_by='column', auto_adjust=True)
close = raw["Close"]
print(close)
