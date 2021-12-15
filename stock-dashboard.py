from time import time
from urllib.request import Request, urlopen
from bs4.element import ResultSet
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit.elements import altair
from streamlit.proto.Empty_pb2 import Empty
plt.style.use('seaborn-dark')
st.set_page_config(layout="wide")

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import yfinance as yf
import yahoo_fin.stock_info as si
import requests
import bs4 as bs
import pickle
import talib

import base64
base64.decodestring = base64.decodebytes

import pygooglenews

# from patterns import patterns

gn = pygooglenews.GoogleNews(lang='en', country='US')

st.title("Stock Analysis and Prediction Dashboard")

def home_page():
    st.header("Home")

    def get_sp500_tickers():
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)
            
        with open("sp500tickers.pickle","wb") as f:
            pickle.dump(tickers,f)
            
        return tickers

    # sp_500_tickers = get_sp500_tickers()
    # st.write(sp_500_tickers)

    def get_news_articles(topic):
        results = gn.search(topic)
        articles = results['entries']

        return articles

    def display_headlines(articles):
        for article in articles:
            title = article["title"]
            date = article["published"]
            link = article["link"]
            source = article["source"]

            st.markdown(title + " (" + date + ") " + link)

    def popular_stock_display():
        popular_tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT']

        for ticker in popular_tickers:
            data = yf.download(ticker, period='1d', interval='1m')
            open_price = data['Open'][0]
            latest_price = round(data['Adj Close'][-1], 2)

            daily_percent_change = round((latest_price - open_price)/open_price * 100, 2)
            daily_return = round(latest_price - open_price, 2)

            if ticker == '^GSPC':
                st.markdown('**S&P 500 (^GSPC)**')
            elif ticker == '^DJI':
                st.markdown('**Dow Jones Industrial Average (^DJI)**')
            elif ticker == '^IXIC':
                st.markdown('**NASDAQ Composite (^IXIC)**')
            else:
                st.markdown('**Russell 2000 (^RUT)**')

            latest_price = str(latest_price)
            daily_return = str(daily_return)
            daily_percent_change = str(daily_percent_change)
            st.markdown('$' + latest_price)
            # st.metric(label="Price", value=latest_price, delta=daily_return + " (" + daily_percent_change + "%)")
            st.markdown(daily_return + " (" + daily_percent_change + "%)")

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index, 
                open=data['Open'], 
                high=data['High'], 
                low=data['Low'], 
                close=data['Close'], 
                name="market data"
            ))
                
            fig.update_layout(
                title="Today's Chart",
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig)
            # st.area_chart(chart_data)

    st.subheader("**Popular Stocks**")
    popular_stock_display()

    stock_news = get_news_articles('Stocks')
    st.subheader("**Latest Stock Market Headlines**")
    display_headlines(stock_news)

def stock_page():
    st.header("Stock Information Page")

    def get_stock_data(ticker, time):
        data = None
        if time == "1 Day":
            data = yf.download(ticker, period="1d", interval="1m")
        if time == "5 Day":
            data = yf.download(ticker, period="5d", interval="15m")
        if time == "1 Month":
            data = yf.download(ticker, period="1mo", interval="1h")
        if time == "3 Month":
            data = yf.download(ticker, period="3mo", interval="1d")
        if time == "6 Month":
            data = yf.download(ticker, period="6mo", interval="1d")
        if time == "Year to Date":
            data = yf.download(ticker, period="ytd", interval="1d")
        if time == "1 Year":
            data = yf.download(ticker, period="1y", interval="1d")
        if time == "2 Year":
            data = yf.download(ticker, period="2y", interval="1w")
        if time == "All Time":
            data = yf.download(ticker, period="max", interval="1mo")
        return data

    def get_company_name(ticker):
        # url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(ticker)
        # result = requests.get(url).json()

        # for x in result['ResultSet']['Result']:
        #     if x['symbol'] == ticker:
        #         return x['company']
        company = yf.Ticker(ticker)
        name = company.info['shortName']
        return name
    
    def matplotlib_chart(data):
        fig, ax = plt.subplots()

        close_min = min(data["Adj Close"])
        close_max = max(data["Adj Close"])

        ax.plot(data["Adj Close"], color='blue')
        plt.xlabel("Date")
        plt.ylabel("Closing Price (USD)")     
        plt.ylim([close_min*0.8, close_max*1.2])  

        return fig

    def plotly_area_chart(data):
        fig = go.Figure()

        min_max_difference = max(data["Adj Close"]) - min(data["Adj Close"])

        chart_min = min(data["Adj Close"]) - min_max_difference*0.7

        if chart_min < 0:
            chart_min = 0
        chart_max = max(data["Adj Close"]) + min_max_difference*0.3

        if data["Adj Close"][-1] > data["Adj Close"][0]:
            chart_color = 'green'
        else:
            chart_color = 'red'

        fig.add_trace(go.Scatter(x=data.index, y=data["Adj Close"], fillcolor=chart_color, fill='tozeroy', mode='none'))
        fig.update_yaxes(range=[chart_min, chart_max])

        return fig

    def ploly_candlestick_chart(data):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, 
            open=data['Open'], 
            high=data['High'], 
            low=data['Low'], 
            close=data['Close'], 
            name="market data"
        ))
                
        fig.update_layout(
            title="Today's Chart",
            yaxis_title="Stock Price (USD)"
        )

        return fig

    def show_stock_info(ticker):
        quote = si.get_quote_table(ticker)
        info = yf.Ticker(ticker).info

        col1, col2, col3 = st.beta_columns(3)
        
        with col1:
            st.write("Open: **" + str(quote["Open"]) + "**")
            st.write("Previous Close: **" + str(quote["Previous Close"]) + "**")
            st.write("Day's Range: **" + quote["Day's Range"] + "**")
            st.write("52 Week Range: **" + quote["52 Week Range"] + "**")
            st.write("Volume: **" + str(quote["Volume"]) + "**")
            st.write("Average Volume: **" + str(quote["Avg. Volume"]) + "**")
        with col2:
            st.write("Market Cap: **" + str(quote["Market Cap"]) + "**")
            st.write("Beta: **" + str(quote["Beta (5Y Monthly)"]) + "**")
            st.write("EPS: **" + str(quote["EPS (TTM)"]) + "**")
            st.write("P/E Ratio: **" + str(quote["PE Ratio (TTM)"]) + "**")
            st.write("PEG Ratio: **" + str(info["pegRatio"]) + "**")
            st.write("Dividend Ratio: **" + str(quote["Forward Dividend & Yield"]) + "**")
        with col3:
            st.write("PB Ratio: **" + str(info["priceToBook"]) + "**")
            st.write("Profit Margin: **" + str(info["profitMargins"]) + "**")
            st.write("Return on Equity: **" + str(info["returnOnEquity"]) + "**")
            st.write("Current Ratio: **" + str(info["currentRatio"]) + "**")
            st.write("Quote Price: **" + str(quote["Quote Price"]) + "**")
            st.write("Earnings Date: **" + quote["Earnings Date"] + "**")

    def get_company_news(company_name):
        results = gn.search(company_name)
        articles = results['entries']

        for article in articles:
            title = article["title"]
            date = article["published"]
            link = article["link"]
            source = article["source"]

            st.markdown(title + " (" + date + ") " + link)
        
    def get_stocktwits_feed(ticker):
        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json")
        data = r.json()

        for message in data["messages"]:
            st.image(message['user']['avatar_url'])
            st.write(message['user']['username'])
            st.write(message['body'])

    def get_insider(ticker):
        try:
            url = ("http://finviz.com/quote.ashx?t=" + ticker.lower())
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            html = bs.BeautifulSoup(webpage, "html.parser")
            insider = pd.read_html(str(html), attrs = {'class': 'body-table'})[0]
            
            # Clean up insider dataframe
            insider = insider.iloc[1:]
            insider.columns = ['Trader', 'Relationship', 'Date', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total', 'SEC Form 4']
            insider = insider[['Date', 'Trader', 'Relationship', 'Transaction', 'Cost', '# Shares', 'Value ($)', '# Shares Total', 'SEC Form 4']]
            insider = insider.set_index('Date')
            return insider

        except Exception as e:
            return e
        
    st.write("What ticker do you want to look at?")
    ticker = st.text_input(label="Ticker", max_chars=5)
    time = st.selectbox(
        label="Time Period", 
        options=(
            "1 Day",
            "5 Day",
            "1 Month",
            "3 Month",
            "6 Month",
            "Year to Date",
            "1 Year",
            "2 Year",
            "All Time"
        )
    )
    stock_data = get_stock_data(ticker, time)

    if not stock_data.empty:
        company_name = get_company_name(ticker)
        st.subheader(company_name)
        st.dataframe(stock_data)
        st.plotly_chart(plotly_area_chart(stock_data))
        show_stock_info(ticker)
        with st.beta_expander(company_name + " News Headlines"):
            get_company_news(company_name)
        with st.beta_expander("StockTwits Feed"):
            get_stocktwits_feed(ticker)
        with st.beta_expander("Inside Trading Activity"):
            st.table(get_insider(ticker))
    else:
        st.error("This is not a valid stock ticker. Please enter a valid ticker symbol.")

def technical_screener():
    patterns = {
        'CDL2CROWS':'Two Crows',
        'CDL3BLACKCROWS':'Three Black Crows',
        'CDL3INSIDE':'Three Inside Up/Down',
        'CDL3LINESTRIKE':'Three-Line Strike',
        'CDL3OUTSIDE':'Three Outside Up/Down',
        'CDL3STARSINSOUTH':'Three Stars In The South',
        'CDL3WHITESOLDIERS':'Three Advancing White Soldiers',
        'CDLABANDONEDBABY':'Abandoned Baby',
        'CDLADVANCEBLOCK':'Advance Block',
        'CDLBELTHOLD':'Belt-hold',
        'CDLBREAKAWAY':'Breakaway',
        'CDLCLOSINGMARUBOZU':'Closing Marubozu',
        'CDLCONCEALBABYSWALL':'Concealing Baby Swallow',
        'CDLCOUNTERATTACK':'Counterattack',
        'CDLDARKCLOUDCOVER':'Dark Cloud Cover',
        'CDLDOJI':'Doji',
        'CDLDOJISTAR':'Doji Star',
        'CDLDRAGONFLYDOJI':'Dragonfly Doji',
        'CDLENGULFING':'Engulfing Pattern',
        'CDLEVENINGDOJISTAR':'Evening Doji Star',
        'CDLEVENINGSTAR':'Evening Star',
        'CDLGAPSIDESIDEWHITE':'Up/Down-gap side-by-side white lines',
        'CDLGRAVESTONEDOJI':'Gravestone Doji',
        'CDLHAMMER':'Hammer',
        'CDLHANGINGMAN':'Hanging Man',
        'CDLHARAMI':'Harami Pattern',
        'CDLHARAMICROSS':'Harami Cross Pattern',
        'CDLHIGHWAVE':'High-Wave Candle',
        'CDLHIKKAKE':'Hikkake Pattern',
        'CDLHIKKAKEMOD':'Modified Hikkake Pattern',
        'CDLHOMINGPIGEON':'Homing Pigeon',
        'CDLIDENTICAL3CROWS':'Identical Three Crows',
        'CDLINNECK':'In-Neck Pattern',
        'CDLINVERTEDHAMMER':'Inverted Hammer',
        'CDLKICKING':'Kicking',
        'CDLKICKINGBYLENGTH':'Kicking - bull/bear determined by the longer marubozu',
        'CDLLADDERBOTTOM':'Ladder Bottom',
        'CDLLONGLEGGEDDOJI':'Long Legged Doji',
        'CDLLONGLINE':'Long Line Candle',
        'CDLMARUBOZU':'Marubozu',
        'CDLMATCHINGLOW':'Matching Low',
        'CDLMATHOLD':'Mat Hold',
        'CDLMORNINGDOJISTAR':'Morning Doji Star',
        'CDLMORNINGSTAR':'Morning Star',
        'CDLONNECK':'On-Neck Pattern',
        'CDLPIERCING':'Piercing Pattern',
        'CDLRICKSHAWMAN':'Rickshaw Man',
        'CDLRISEFALL3METHODS':'Rising/Falling Three Methods',
        'CDLSEPARATINGLINES':'Separating Lines',
        'CDLSHOOTINGSTAR':'Shooting Star',
        'CDLSHORTLINE':'Short Line Candle',
        'CDLSPINNINGTOP':'Spinning Top',
        'CDLSTALLEDPATTERN':'Stalled Pattern',
        'CDLSTICKSANDWICH':'Stick Sandwich',
        'CDLTAKURI':'Takuri (Dragonfly Doji with very long lower shadow)',
        'CDLTASUKIGAP':'Tasuki Gap',
        'CDLTHRUSTING':'Thrusting Pattern',
        'CDLTRISTAR':'Tristar Pattern',
        'CDLUNIQUE3RIVER':'Unique 3 River',
        'CDLUPSIDEGAP2CROWS':'Upside Gap Two Crows',
        'CDLXSIDEGAP3METHODS':'Upside/Downside Gap Three Methods'
    }

    def screen_stocks(pattern):
        tickers = si.tickers_sp500()
        i = 0
        progress_bar = st.progress(0.0)
        for ticker in tickers:
            i += 1
            progress_bar.progress(i/505)
            try:
                print(f"getting {ticker} data ({i}/505)")
                data = yf.download(ticker, period="1mo", interval="1d")

                pattern_function = getattr(talib, pattern)
                result = pattern_function(data['Open'], data['High'], data['Low'], data['Close'])
                last = result.tail(1).values[0]

                if last > 0:
                    label = 'bullish'
                elif last < 0:
                    label = 'bearish'
                else:
                    label = None

                if label == None:
                    pass
                else:
                    company = yf.Ticker(ticker)
                    name = company.info['shortName']

                    if label == 'bullish':
                        st.write(name + " (" + ticker + ") is Bullish")
                        chart_url = f"https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l"
                        st.image(chart_url)
                    elif label == 'bearish':
                        st.write(name + " (" + ticker + ") is Bearish")
                        chart_url = f"https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l"
                        st.image(chart_url)
                
            except:
                pass

    st.header("S&P 500 Technical Screener")
    # st.write(patterns)
    pattern_name = st.selectbox('What pattern do you want to screen for', list(patterns.values()))
    pattern_value = list(patterns.keys())[list(patterns.values()).index(pattern_name)]
    screen_stocks(pattern_value)

def technical_analysis_and_stock_projection():
    def overlap_studies(ticker):
        indicators = {
            'BBANDS':'Bollinger Bands',
            'DEMA':'Double Exponential Moving Average',
            'EMA':'Exponential Moving Average',
            'HT_TRENDLINE':'Hilbert Transform - Instantaneous Trendline',
            'KAMA':'Kaufman Adaptive Moving Average',
            'MA':'Moving average',
            'MAMA':'MESA Adaptive Moving Average',
            'MAVP':'Moving average with variable period',
            'MIDPOINT':'MidPoint over period',
            'MIDPRICE':'Midpoint Price over period',
            'SAR':'Parabolic SAR',
            'SAREXT':'Parabolic SAR - Extended',
            'SMA':'Simple Moving Average',
            'T3':'Triple Exponential Moving Average (T3)',
            'TEMA':'Triple Exponential Moving Average',
            'TRIMA':'Triangular Moving Average',
            'WMA':'Weighted Moving Average'
        }
        indicator_name = st.selectbox("Which indicator do you want to see?", list(indicators.values()))
        indicator_value = list(indicators.keys())[list(indicators.values()).index(indicator_name)]
        indicator_function = getattr(talib, indicator_value)
        
        data = yf.download(ticker, period="6mo", interval="1d")
        company = yf.Ticker(ticker)
        company_name = company.info['shortName']
        df = data

        if indicator_value == "BBANDS":
            upper, mid, lower = indicator_function(data['Adj Close'])
            df["Upper"] = upper
            df["Middle"] = mid
            df["Lower"] = lower

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ))

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['Upper'],
                line_color = 'yellow',
                name = 'Upper Band',
                opacity = 0.5
            ))

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['Middle'],
                line_color = 'turquoise',
                name = 'Middle Band',
                opacity = 0.5
            ))
            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['Lower'],
                line_color = 'purple',
                name = 'Lower Band',
                opacity = 0.5
            ))
            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "MAVP":
            df["periods"] = np.arange(len(df))
            df["real"] = indicator_function(df["Adj Close"], df["periods"])

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ))

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ))

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "MAMA":
            df["MAMA"], df["FAMA"] = indicator_function(data["Adj Close"])
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ))

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['MAMA'],
                line_color = 'yellow',
                name = 'MAMA',
                opacity = 0.5
            ))

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['FAMA'],
                line_color = 'purple',
                name = 'FAMA',
                opacity = 0.5
            ))

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "MIDPRICE" or indicator_value == "SAR" or indicator_value == "SAREXT":
            df["real"] = indicator_function(df["High"], df["Low"])

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ))

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ))

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            df["real"] = indicator_function(df["Adj Close"])

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ))

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ))

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
    
    def momentum_indicators(ticker):
        indicators = {
            'ADX':'Average Directional Movement Index',
            'ADXR':'Average Directional Movement Index Rating',
            'APO':'Absolute Price Oscillator',
            'AROON':'Aroon',
            'AROONOSC':'Aroon Oscillator',
            'BOP':'Balance Of Power',
            'CCI':'Commodity Channel Index',
            'CMO':'Chande Momentum Oscillator',
            'DX':'Directional Movement Index',
            'MACD':'Moving Average Convergence/Divergence',
            'MACDEXT':'MACD with controllable MA type',
            'MACDFIX':'Moving Average Convergence/Divergence Fix 12/26',
            'MFI':'Money Flow Index',
            'MINUS_DI':'Minus Directional Indicator',
            'MINUS_DM':'Minus Directional Movement',
            'MOM':'Momentum',
            'PLUS_DI':'Plus Directional Indicator',
            'PLUS_DM':'Plus Directional Movement',
            'PPO':'Percentage Price Oscillator',
            'ROC':'Rate of change : ((price/prevPrice)-1)*100',
            'ROCP':'Rate of change Percentage: (price-prevPrice)/prevPrice',
            'ROCR':'Rate of change ratio: (price/prevPrice)',
            'ROCR100':'Rate of change ratio 100 scale: (price/prevPrice)*100',
            'RSI':'Relative Strength Index',
            'STOCH':'Stochastic',
            'STOCHF':'Stochastic Fast',
            'STOCHRSI':'Stochastic Relative Strength Index',
            'TRIX':'1-day Rate-Of-Change (ROC) of a Triple Smooth EMA',
            'ULTOSC':'Ultimate Oscillator',
            'WILLR':'Williams %R'
        }
        
        indicator_name = st.selectbox("Which indicator do you want to see?", list(indicators.values()))
        indicator_value = list(indicators.keys())[list(indicators.values()).index(indicator_name)]
        indicator_function = getattr(talib, indicator_value)
        
        data = yf.download(ticker, period="6mo", interval="1d")
        company = yf.Ticker(ticker)
        company_name = company.info['shortName']
        df = data

        if indicator_value == "AROON":
            df["Aroon Down"], df["Aroon Up"] = indicator_function(df["High"], df["Low"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )

            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ),row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['Aroon Up'],
                line_color = 'yellow',
                name = 'Aroon Up',
                opacity = 0.5
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['Aroon Down'],
                line_color = 'purple',
                name = 'Aroon Down',
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "MACD" or indicator_value == "MACDEXT" or indicator_value == "MACDFIX":
            df["macd"], df["macdsignal"], df["macdhist"] = indicator_function(df["Adj Close"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['macd'],
                line_color = 'yellow',
                name = 'MACD',
                opacity = 0.5
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['macdsignal'],
                line_color = 'purple',
                name = 'MACD Signal',
                opacity = 0.5
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['macdhist'],
                line_color = 'turquoise',
                name = 'MACD History',
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "STOCH" or indicator_value == "STOCHF":
            df["k"], df["d"] = indicator_function(df["High"], df["Low"], df["Adj Close"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )

            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ),row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['k'],
                line_color = 'yellow',
                name = '%K',
                opacity = 0.5
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['d'],
                line_color = 'purple',
                name = '%D',
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "STOCHRSI":
            df["k"], df["d"] = indicator_function(df["Adj Close"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )

            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ),row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['k'],
                line_color = 'yellow',
                name = '%K',
                opacity = 0.5
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['d'],
                line_color = 'purple',
                name = '%D',
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "APO" or indicator_value == "CMO" or indicator_value == "MOM" or indicator_value == "PPO" or indicator_value == "ROC" or indicator_value == "ROCP" or indicator_value == "ROCP" or indicator_value == "ROCR100" or indicator_value == "RSI" or indicator_value == "TRIX":
            df["real"] = indicator_function(df["Adj Close"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ), row=2, col=1)
            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "BOP":
            df["real"] = indicator_function(df["Open"], df["High"], df["Low"], df["Adj Close"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "MFI":
            df["real"] = indicator_function(df["High"], df["Low"], df["Close"], df["Volume"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        elif indicator_value == "AROONOSC" or indicator_value == "MINUS_DM":
            df["real"] = indicator_function(df["High"], df["Low"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            df["real"] = indicator_function(df["High"], df["Low"], df["Adj Close"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_value,
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
    
    def volatility_indicators(ticker):
        indicators = {
            'AD':'Chaikin A/D Line',
            'ADOSC':'Chaikin A/D Oscillator',
            'OBV':'On Balance Volume'
        }
        
        indicator_name = st.selectbox("Which indicator do you want to see?", list(indicators.values()))
        indicator_value = list(indicators.keys())[list(indicators.values()).index(indicator_name)]
        indicator_function = getattr(talib, indicator_value)
        
        data = yf.download(ticker, period="6mo", interval="1d")
        company = yf.Ticker(ticker)
        company_name = company.info['shortName']
        df = data

        if indicator_value == "OBV":
            df["real"] = indicator_function(df["Adj Close"], df["Volume"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_name,
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            df["real"] = indicator_function(df["High"], df["Low"], df["Close"], df["Volume"])

            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
            )
            fig.add_trace(go.Candlestick(x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Adj Close'],
                name = f'{company_name} Data'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['real'],
                line_color = 'yellow',
                name = indicator_name,
                opacity = 0.5
            ), row=2, col=1)

            fig.update_layout(
                title=indicator_name,
                yaxis_title="Stock Price (USD)"
            )

            st.plotly_chart(fig, use_container_width=True)
    
    st.header("Stock Projection and Technical Analysis")
    ticker = st.text_input("Stock Ticker", max_chars=5)
    analysis = st.selectbox("What type of technical analysis would you like to do?", ("Overlap Studies", "Momentum Indicators", "Volatility Indicators"))

    if analysis == "Overlap Studies":
        overlap_studies(ticker)
    if analysis == "Momentum Indicators":
        momentum_indicators(ticker)
    if analysis == "Volatility Indicators":
        volatility_indicators(ticker)


# home_page()
# stock_page()
# technical_screener()
technical_analysis_and_stock_projection()