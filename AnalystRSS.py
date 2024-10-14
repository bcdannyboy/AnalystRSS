#!/usr/bin/env python

"""
Analyst Prediction Analyzer Utility
analyze the analysts then analyze their analysis

This utility fetches analyst price targets and stock prices to evaluate the prediction accuracy
of financial analysts. It analyzes the analysts' accuracy once a week and identifies the top 10
analysts based on their historical performance. It also generates and updates an RSS feed every
12 hours with the latest reports from the top analysts.

Usage:
    python AnalystRSS.py --symbols symbols.txt
"""

import os
import sys
import argparse
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import schedule
import time
from dotenv import load_dotenv
from feedgen.feed import FeedGenerator
import urllib.parse

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("FMP_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[
                        logging.FileHandler("analyst_prediction_analyzer.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Global variable for command-line arguments
args = None

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Analyst Prediction Analyzer")
    parser.add_argument('--symbols', required=True, help='Path to the symbols txt file')
    return parser.parse_args()

def read_symbols(symbols_file):
    """
    Reads stock symbols from a text file.

    Parameters:
    symbols_file (str): Path to the symbols file.

    Returns:
    list: List of stock symbols.
    """
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        return symbols
    except Exception as e:
        logger.error(f"Error reading symbols file {symbols_file}: {e}")
        return []

def get_latest_price_targets(analyst_name):
    """
    Fetches the latest price targets for a given analyst from the Financial Modeling Prep API.

    Parameters:
    analyst_name (str): The name of the analyst.

    Returns:
    list: A list of the latest price target alerts for the analyst.
    """
    try:
        encoded_name = urllib.parse.quote(analyst_name)
        url = f"https://financialmodelingprep.com/api/v4/price-target-analyst-name?name={encoded_name}&apikey={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                # Filter data to only those within the last month
                one_month_ago = datetime.utcnow() - timedelta(days=30)
                data = [item for item in data if dateparser.parse(item['publishedDate']) >= one_month_ago]
                if not data:
                    return None
                # Sort data by publishedDate descending
                data.sort(key=lambda x: x['publishedDate'], reverse=True)
                # For each symbol, keep only the latest alert
                latest_alerts = {}
                for item in data:
                    symbol = item['symbol']
                    if symbol not in latest_alerts:
                        latest_alerts[symbol] = item
                # Return the list of latest alerts per ticker
                return list(latest_alerts.values())
            else:
                return None
        else:
            logger.error(f"Error fetching data for {analyst_name}: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception in get_latest_price_targets for {analyst_name}: {e}")
        return None

def get_current_stock_price(symbol):
    """
    Fetches the current stock price for a given symbol using yfinance.

    Parameters:
    symbol (str): The stock symbol.

    Returns:
    float: The current stock price, or None if not found.
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='1d')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            return current_price
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching current stock price for {symbol}: {e}")
        return None

def get_price_targets_for_symbols(symbols):
    """
    Fetches price targets for a list of symbols from the Financial Modeling Prep API.

    Parameters:
    symbols (list): List of stock symbols.

    Returns:
    pandas.DataFrame: DataFrame containing price targets for the symbols.
    """
    all_price_targets = []
    for symbol in symbols:
        try:
            url = f"https://financialmodelingprep.com/api/v4/price-target?symbol={symbol}&apikey={API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    price_target_df = pd.DataFrame(data)
                    price_target_df["symbol"] = symbol
                    all_price_targets.append(price_target_df)
            else:
                logger.error(f"Error fetching price target data for {symbol}: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception in get_price_targets_for_symbols for {symbol}: {e}")
    if all_price_targets:
        all_price_target_df = pd.concat(all_price_targets, ignore_index=True)
        return all_price_target_df
    else:
        return pd.DataFrame()

def get_historical_data(symbol):
    """
    Fetches historical stock price data for a given symbol using yfinance.

    Parameters:
    symbol (str): The stock symbol.

    Returns:
    pandas.Series: Series containing historical 'Close' prices, or None if not found.
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="max")
        if not hist.empty:
            return hist['Close']
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def analyze_analyst_skill(price_targets_df, symbols, output_file='analyst_accuracy.csv', time_horizon_days=90):
    """
    Analyzes analysts' predictions and updates accuracy data.

    Parameters:
    price_targets_df (DataFrame): DataFrame containing price targets.
    symbols (list): List of stock symbols.
    output_file (str): Path to the output CSV file.
    time_horizon_days (int): Time horizon in days to evaluate predictions.
    """
    logger.info("Analyzing analyst skill...")

    # Ensure that price_targets_df has the necessary columns
    if price_targets_df.empty or 'publishedDate' not in price_targets_df.columns:
        logger.error("No price target data to analyze.")
        return

    # Convert publishedDate to datetime and set as index
    price_targets_df['publishedDate'] = pd.to_datetime(price_targets_df['publishedDate']).dt.normalize()
    price_targets_df.set_index('publishedDate', inplace=True)

    # Fetch historical data for symbols
    logger.info("Fetching historical data for symbols...")
    all_hist_data = {}
    for symbol in symbols:
        hist_data = get_historical_data(symbol)
        if hist_data is not None:
            hist_data.index = hist_data.index.normalize()
            all_hist_data[symbol] = hist_data

    # Prepare data for analysis
    analyst_skill = {}

    # Analyze each symbol
    for symbol in symbols:
        if symbol not in all_hist_data:
            continue

        hist_data = all_hist_data[symbol]
        symbol_price_targets = price_targets_df[price_targets_df['symbol'] == symbol]

        if symbol_price_targets.empty:
            continue

        # For each price target, compute the error after the time horizon
        errors = []
        for idx, row in symbol_price_targets.iterrows():
            published_date = idx
            target_date = published_date + timedelta(days=time_horizon_days)

            # Find the closest available date in historical data
            actual_dates = hist_data.index
            if target_date > actual_dates[-1]:
                # Cannot compute error if target date is beyond available historical data
                continue
            idx_closest = actual_dates.get_indexer([target_date], method='nearest')[0]
            if idx_closest == -1:
                continue  # Skip if no nearest date is found
            actual_date = actual_dates[idx_closest]

            actual_price = hist_data.loc[actual_date]

            # Compute percentage error
            predicted_price = row["priceTarget"]
            if predicted_price == 0:
                continue  # Avoid division by zero
            percentage_error = abs((actual_price - predicted_price) / predicted_price) * 100
            errors.append({
                "analystName": row["analystName"],
                "analystCompany": row["analystCompany"],
                "publisher": row["newsPublisher"],
                "symbol": symbol,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "percentage_error": percentage_error
            })

        if not errors:
            continue

        errors_df = pd.DataFrame(errors)

        # Aggregate errors per analyst
        for name, group in errors_df.groupby("analystName"):
            avg_error = group["percentage_error"].mean()
            count = len(group)
            diversity = group["symbol"].nunique()
            breadth = count * diversity  # Number of predictions times number of unique symbols
            accuracy = 1 / avg_error if avg_error != 0 else 0  # Avoid division by zero
            skill_score = accuracy * breadth  # Skill is breadth x accuracy

            if name in analyst_skill:
                analyst_skill[name]["breadth"] += breadth
                analyst_skill[name]["total_predictions"] += count
                analyst_skill[name]["total_error"] += group["percentage_error"].sum()
                analyst_skill[name]["unique_symbols"].update(group["symbol"].unique())
            else:
                analyst_skill[name] = {
                    "analystCompany": group["analystCompany"].iloc[0],
                    "publisher": group["publisher"].iloc[0],
                    "breadth": breadth,
                    "total_predictions": count,
                    "total_error": group["percentage_error"].sum(),
                    "unique_symbols": set(group["symbol"].unique())
                }

    # Calculate final skill score for each analyst
    for name in analyst_skill:
        total_error = analyst_skill[name]["total_error"]
        total_predictions = analyst_skill[name]["total_predictions"]
        avg_error = total_error / total_predictions if total_predictions != 0 else float('inf')
        accuracy = 1 / avg_error if avg_error != 0 else 0
        breadth = analyst_skill[name]["breadth"]
        skill_score = accuracy * breadth
        analyst_skill[name]["avg_error"] = avg_error
        analyst_skill[name]["accuracy"] = accuracy
        analyst_skill[name]["skill_score"] = skill_score
        analyst_skill[name]["diversity"] = len(analyst_skill[name]["unique_symbols"])

    # Convert analyst_skill to DataFrame
    analyst_skill_df = pd.DataFrame.from_dict(analyst_skill, orient='index')
    analyst_skill_df.reset_index(inplace=True)
    analyst_skill_df.rename(columns={'index': 'analystName'}, inplace=True)

    # Sort analysts by skill score
    analyst_skill_df.sort_values(by='skill_score', ascending=False, inplace=True)

    # Write the results to the output file
    columns_to_output = [
        'analystName', 'analystCompany', 'publisher', 'skill_score',
        'accuracy', 'avg_error', 'breadth', 'total_predictions', 'diversity'
    ]
    analyst_skill_df.to_csv(output_file, columns=columns_to_output, index=False)

    logger.info(f"Analyst skill scores have been written to {output_file}")

def find_top_analysts(input_csv, top_n=10, min_predictions=5):
    """
    Identifies the top N analysts based on accuracy and prediction volume.

    Parameters:
    input_csv (str): Path to the input CSV file containing analysts' accuracy data.
    top_n (int): Number of top analysts to return.
    min_predictions (int): Minimum number of predictions an analyst must have made to be considered.

    Returns:
    pandas.DataFrame: DataFrame containing the top N analysts.
    """
    try:
        df = pd.read_csv(input_csv)

        # Handle infinite or NaN accuracy values (due to zero avg_error)
        df.replace([float('inf'), float('nan')], 0, inplace=True)

        # Filter out analysts with fewer than min_predictions
        df_filtered = df[df['total_predictions'] >= min_predictions]

        if df_filtered.empty:
            logger.warning("No analysts meet the minimum prediction requirement.")
            return pd.DataFrame()

        # Calculate the weighted score: accuracy multiplied by total predictions
        df_filtered['weighted_score'] = df_filtered['accuracy'] * df_filtered['total_predictions']

        # Sort the DataFrame based on weighted_score in descending order
        df_sorted = df_filtered.sort_values(by='weighted_score', ascending=False)

        # Select the top N analysts
        top_analysts = df_sorted.head(top_n)

        # Log the top analysts
        logger.info("Top Analysts based on accuracy and volume:")
        logger.info(top_analysts[['analystName', 'analystCompany', 'publisher', 'accuracy', 'avg_error', 'total_predictions', 'diversity', 'weighted_score']])

        return top_analysts

    except Exception as e:
        logger.error(f"Error in find_top_analysts: {e}")
        return pd.DataFrame()

def update_rss_feed():
    """
    Generates or updates the RSS feed with the latest reports from the top analysts.
    """
    logger.info("Updating RSS feed...")

    # Identify the top analysts
    top_analysts = find_top_analysts('analyst_accuracy.csv')
    if top_analysts.empty:
        logger.error("No top analysts to generate RSS feed.")
        return

    # Fetch the latest reports from the top analysts
    # For each top analyst, fetch their latest price targets
    feed_entries = []
    for idx, row in top_analysts.iterrows():
        analyst_name = row['analystName']
        latest_alerts = get_latest_price_targets(analyst_name)
        if latest_alerts:
            for alert in latest_alerts:
                # Build the RSS feed entry
                feed_entry = {
                    'analystName': analyst_name,
                    'analystCompany': row['analystCompany'],
                    'accuracy': row['accuracy'],
                    'total_predictions': row['total_predictions'],
                    'newsTitle': alert.get('newsTitle', ''),
                    'newsURL': alert.get('newsURL', ''),
                    'publishedDate': alert.get('publishedDate', ''),
                    'description': alert.get('newsTitle', '')
                }
                feed_entries.append(feed_entry)
        else:
            logger.warning(f"No recent alerts for analyst {analyst_name}")

    # Generate the RSS feed
    if feed_entries:
        fg = FeedGenerator()
        fg.title('Top Analysts Reports')
        fg.link(href='http://example.com/rss', rel='self')
        fg.description('Latest reports from top financial analysts')

        for entry in feed_entries:
            fe = fg.add_entry()
            fe.title(f"{entry['analystName']} - {entry['newsTitle']}")
            fe.link(href=entry['newsURL'])
            try:
                pub_date = dateparser.parse(entry['publishedDate'])
            except Exception:
                pub_date = datetime.utcnow()
            fe.pubDate(pub_date)
            fe.description(f"Analyst: {entry['analystName']}<br>"
                           f"Company: {entry['analystCompany']}<br>"
                           f"Accuracy: {entry['accuracy']}<br>"
                           f"Total Predictions: {entry['total_predictions']}<br>"
                           f"Report: {entry['description']}")

        fg.rss_file('top_analysts_feed.xml')
        logger.info("RSS feed updated.")
    else:
        logger.warning("No feed entries to generate RSS feed.")

def weekly_analysis():
    """
    Performs weekly analysis of analysts' predictions.
    Fetches latest price targets, compares to actual stock prices,
    updates analysts' accuracy data, and identifies top analysts.
    """
    logger.info("Starting weekly analysis...")

    # Read symbols from the symbols file
    symbols = read_symbols(args.symbols)
    if not symbols:
        logger.error("No symbols to process.")
        return

    # Fetch price targets for the symbols
    logger.info("Fetching price targets for symbols...")
    price_targets_df = get_price_targets_for_symbols(symbols)
    if price_targets_df.empty:
        logger.error("No price target data fetched.")
        return

    # Analyze analysts' predictions
    logger.info("Analyzing analysts' predictions...")
    analyze_analyst_skill(price_targets_df, symbols)

    logger.info("Weekly analysis completed.")

def main():
    """
    Main function to set up scheduling and start the utility.
    """
    global args
    args = parse_arguments()

    # Check if API key is available
    if not API_KEY:
        logger.error("API Key not found. Please set FMP_API_KEY in .env file.")
        sys.exit(1)

    # Perform initial analysis and RSS feed update
    logger.info("Performing initial analysis and RSS feed update...")
    weekly_analysis()
    update_rss_feed()

    # Schedule weekly_analysis() every Monday at 9 AM
    schedule.every().monday.at("09:00").do(weekly_analysis)

    # Schedule update_rss_feed() every 12 hours
    schedule.every(12).hours.do(update_rss_feed)

    logger.info("Starting the Analyst Prediction Analyzer...")

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
