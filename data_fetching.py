# Sourcing the data for all the stocks of S&P500
# Load libraries
import pandas as pd
import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests
import csv
# Yahoo for dataReader
import yfinance as yf
yf.pdr_override()

import warnings
warnings.filterwarnings('ignore')

# Load dataset
# Scraping wikipedia to fetch S&P 500 stock list

def save_tickers():
    DJIAurl = "https://www.dividendmax.com/market-index-constituents/dow-jones-30"
    request_DJIA = requests.get(DJIAurl)
    soup = bs.BeautifulSoup(request_DJIA.text, 'html.parser')
    table = soup.find('table', {'class': 'mdc-data-table__table'})
    header = table.findAll("th")
    if header[1].text.rstrip() != "Ticker":
        raise Exception("Can't parse website's table!")
        # Retrieve the values in the table
    tickers = []
    rows = table.findAll("tr")
    for row in rows:
        fields = row.findAll("td")
        if fields:
            ticker = fields[1].text.rstrip()
            tickers.append(str(ticker))
    print('Tickers \n', tickers)
    return tickers


save_tickers()


def get_data_from_yahoo():
    tickers = save_tickers()
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()
    dataset = yf.download(tickers, start=start, end=end)['Adj Close']
    dataset.to_csv("DJIAData.csv")
    return dataset.to_csv("DJIAData.csv")


get_data_from_yahoo()
