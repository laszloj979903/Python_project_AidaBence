import streamlit as st
import requests
import pandas as pd
import datetime
import time
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from finvader import finvader
import yfinance as yf

import nltk
nltk.download('vader_lexicon')

#%%Creating streamlit title and short description
st.title("Stock News Sentiment Analysis")
st.subheader("Final Project: Data Processing in Python  \n Aida Hodzic & Bence Laszlo" )
st.write("Using a FinVADER package specialized in financial lexicon, sentiment scores are assigned to stock tickers on a daily, weekly and monthly basis. Scores are calculated using a combination of article titles and description.  \n Using financial news data from YahooFinance, we also provide plots of the adjusted close prices against the computed sentiment scores.  \n The assigned sentiment scores are presented using WordClouds, time-series plots and filtered tables that provide more context. \n For more information, a new WordCloud in the end is generated only for articles with negative sentiment scores, if there are any.")

#%%Required inputs for sentiment analysis, API key is specific to the news site
api_key = 'OTl2SM9_8xGEkqop_pj57cYyS4gjsurl'
news_url = "https://api.polygon.io/v2/reference/news"

#%%Creating the sidebar for color maps that work for both wordcloud and the trend analysis chart
color_map = {
    "Blues": "blue",
    "Reds": "red",
    "Greens": "green",
    "coolwarm": "cyan",
    "viridis": "yellowgreen",
    "plasma": "orangered",
    "inferno": "darkred",
    "magma": "purple",
    "cividis": "gold"
}
color_scheme = st.sidebar.selectbox("Choose Color Scheme:", list(color_map.keys()))
chart_color = color_map[color_scheme]

st.sidebar.markdown("---")  # Adds a horizontal line for separation

#%%Creating the user input bars for streamlit

ticker = st.sidebar.text_input("Enter the stock ticker symbol:", "AAPL").strip().upper()
date_choice = st.sidebar.selectbox("Choose the date range:", ["previous day", "previous week", "previous month"], index=1)
st.text("We recommend using weekly or monthly horizons for more insights.")
st.sidebar.markdown("---")  # Adds a horizontal line for separation

#%%Determining the date range based on possible inputs 
today = datetime.date.today()
if date_choice == "previous day":
    start_date = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
elif date_choice == "previous week":
    start_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
elif date_choice == "previous month":
    start_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

#%%News macro that fetches news articles from the site based on the ticker input (issues when data is requested too quickly -- sleeper added, also issues when requesting more than 5 tickers per minute -- cannot be solved as higher limits are locked behind pay wall but added error message that shows issue)
def get_news(ticker, start_date, end_date, api_key, news_url, limit=100):
    params = {
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "limit": limit,
        "apiKey": api_key
    }
    time.sleep(random.uniform(1, 3))  
    
    response = requests.get(news_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    else:
        st.error(f"Failed to get news: {response.status_code} - {response.text}")
        return []

#%%Acquiring news articles for the selected ticker (as ticker input is not dropdown -- added warning when no news articles could be found ie wrong ticker)
articles = get_news(ticker, start_date, end_date, api_key, news_url)

if not articles:
    st.warning(f"No news articles found for the selected period for {ticker}.")
else:
    #Save articles in a dataframe --- there is limit for the number of news articles that we can download from the site = 100
    df = pd.DataFrame(articles)[["title", "description", "published_utc"]]
    
#%%FinVADER sentiment analysis on news articles macro
def sentiment_analysis(text):
    try:
        score = finvader(
            text,
            use_sentibignomics=True,  # Use SentiBignomics lexicon
            use_henry=True,          # Use Henry's lexicon
            indicator="compound"     # Compound sentiment score
        )
        return score
    except Exception as e:
        print(f"Error in FinVADER analysis: {e}")
        return None

    
df['combined_text'] = df['title'] + " " + df['description'].fillna("")
df['combined_score'] = df['combined_text'].apply(sentiment_analysis)
    
#%%Aggregate the sentiment analysis data based on date
df['published_utc'] = pd.to_datetime(df['published_utc'])
sentiment_data = df.groupby(df['published_utc'].dt.date)['combined_score'].mean().reset_index()
    
#%%Table that can be filtered for postivie and negative scores
st.subheader(f"News Articles for {ticker}")
if not df.empty and 'combined_score' in df.columns:
    filter_option = st.radio(
        "Filter articles by their sentiment scores:",
        ("All Articles", "Positive Scores", "Negative Scores")
    )
    if filter_option == "Positive Scores":
        filtered_df = df[df['combined_score'] > 0][["title", "description", "combined_score", "published_utc"]]
    elif filter_option == "Negative Scores":
        filtered_df = df[df['combined_score'] < 0][["title", "description", "combined_score", "published_utc"]]
    else:
        filtered_df = df[["title", "description", "combined_score", "published_utc"]]
    st.subheader(f"Articles for {ticker} from {date_choice}")
    st.data_editor(filtered_df, num_rows="fixed")

#%% stooock prices for comparison! 

def fetch_and_plot_stock(ticker, date_choice, sentiment_data):
    period_mapping = {
        "previous day": "1d",
        "previous week": "5d",
        "previous month": "1mo"
    }
    
    stock = yf.Ticker(ticker)
    data = stock.history(period=period_mapping[date_choice], interval="1h" if date_choice == "previous day" else "1d")
    
    if data.empty or sentiment_data.empty:
        st.warning("No data available for the selected period.")
        return
    
    data.reset_index(inplace=True)
    sentiment_data['published_utc'] = pd.to_datetime(sentiment_data['published_utc'])
    daily_avg_sentiment = sentiment_data.groupby(sentiment_data['published_utc'].dt.date)['combined_score'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sentiment_data['published_utc'], y=sentiment_data['combined_score'],
                             mode='markers', marker=dict(color='crimson', size=6),
                             name='Sentiment Score'))
    
    fig.add_trace(go.Scatter(x=daily_avg_sentiment['published_utc'], y=daily_avg_sentiment['combined_score'],
                             mode='lines', line=dict(color=chart_color, width=2),
                             name='Daily Average Sentiment'))

    fig.add_trace(go.Scatter(x=data['Date'], y=data.get('Adj Close', data['Close']),
                             mode='lines', line=dict(color='royalblue', width=2),
                             name='Stock Price', yaxis='y2'))
    
    fig.update_layout(
        title=f"{ticker} Stock Price and Sentiment Score for {date_choice}",
        xaxis_title='Date',
        yaxis=dict(title='Sentiment Score', side='left', color='crimson'),
        yaxis2=dict(title='Stock Price (USD)', side='right', overlaying='y', color='royalblue')
    )
    
    st.plotly_chart(fig)
    st.write("Please note that there might be minor discrepancies due to different timeframes: stock data considers business days, whereas sentiment data spans all days.")

st.subheader("Stock Close Prices vs. Sentiment Score")
fetch_and_plot_stock(ticker, date_choice, sentiment_data)

#%%Including historical sentiment trend chart for tickewr
if not df.empty and 'combined_score' in df.columns:
    st.subheader(f"Sentiment Score trend for {ticker}")

    df['published_utc'] = pd.to_datetime(df['published_utc'])

    daily_avg_sentiment = df.groupby(df['published_utc'].dt.date)['combined_score'].mean().reset_index()

    fig = px.scatter(df, x='published_utc', y='combined_score', title=f"Sentiment Scores for {date_choice}",
                     labels={'combined_score': 'Sentiment Score', 'published_utc': 'Date'},
                     color_discrete_sequence=[chart_color])

    best_articles = df.nlargest(5, 'combined_score') #we did 5 best and worst articles
    worst_articles = df.nsmallest(5, 'combined_score')

    fig.add_scatter(x=best_articles['published_utc'], y=best_articles['combined_score'], 
                    mode='markers', marker=dict(color='green', size=10, symbol='star'), 
                    name='Best Articles')

    fig.add_scatter(x=worst_articles['published_utc'], y=worst_articles['combined_score'], 
                    mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), 
                    name='Worst Articles')

    fig.add_scatter(x=daily_avg_sentiment['published_utc'], y=daily_avg_sentiment['combined_score'], 
                    mode='lines', line=dict(color=chart_color, width=2), 
                    name='Daily Average Sentiment')
    st.plotly_chart(fig)
    
#%%Including descriptive statistics based on the dataframe table
if not df.empty and 'combined_score' in df.columns:
    st.subheader(f"Descriptive Statistics for {ticker} score for {date_choice}")
    st.write("Brief overview for a better understanding of the assigned sentiment values.")

    descriptive_stats = df['combined_score'].describe().to_frame().T
    descriptive_stats.rename(columns={'combined_score': 'Sentiment Score Statistics'}, inplace=True)

    st.dataframe(descriptive_stats)
else:
    st.warning("No data available to perform descriptive statistics.")

#%%Generating WordCloud based on ticker input
text_data = " ".join(df["combined_text"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color_scheme).generate(text_data)
    
st.subheader(f"Word Cloud for {ticker}")
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

#%%Sliced
if not df.empty and 'combined_score' in df.columns:
    best_articles = df.nlargest(5, 'combined_score')[["title", "description", "combined_score", "published_utc"]]
    worst_articles = df.nsmallest(5, 'combined_score')[["title", "description", "combined_score", "published_utc"]]

    st.subheader(f"Articles with Best Sentiment Scores for {ticker}")
    st.data_editor(best_articles, num_rows="dynamic")

    st.subheader(f"Articles with Worst Sentiment Scores for {ticker}")
    st.data_editor(worst_articles, num_rows="dynamic")
    

    
#%%Generating WordCloud based only on poor sentiment news articles (score below 0) based on ticker input
poor_sentiment_data = df[df['combined_score'] < 0]
if not poor_sentiment_data.empty:
        st.subheader(f"Word Cloud for articles with negative sentiment score for {ticker}")
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap=color_scheme).generate(" ".join(poor_sentiment_data['combined_text'].dropna()))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

st.write("If wordcloud brings up an interesting word, try using the search option and looking for it in the first table to dig deeper into the financial news and the context of it.")

