# AnalystRSS

Analyze the analysts, then analyze their analysis.

## Overview

**AnalystRSS** is a Python utility designed to evaluate the prediction accuracy of financial analysts. It fetches analyst price targets and stock prices, analyzes the analysts' historical performance, identifies the top analysts, and generates an RSS feed with their latest reports. This allows users to stay updated with insights from the most accurate financial analysts.

## Features

- **Weekly Analysis**: Fetches the latest price targets and updates analysts' accuracy every Monday at 9 AM.
- **Top Analyst Identification**: Identifies and ranks the top 10 analysts based on their prediction accuracy and volume.
- **RSS Feed Generation**: Generates and updates an RSS feed every 12 hours with the latest reports from the top analysts.
- **Scheduling**: Automates tasks using the `schedule` library for seamless operation.
- **Logging**: Provides detailed logs for monitoring and debugging.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Financial Modeling Prep API Key**

### Required Python Libraries

Ensure you have the following libraries installed:

- `requests`
- `pandas`
- `yfinance`
- `schedule`
- `feedgen`
- `python-dotenv`

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/AnalystRSS.git
   cd AnalystRSS
   ```

2. **Install Required Libraries**

   Install the required Python libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   *Alternatively, install them manually:*

   ```bash
   pip install requests pandas yfinance schedule feedgen python-dotenv
   ```

3. **Set Up API Key**

   - Obtain your API key from [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/).
   - Create a `.env` file in the root directory of the project:

     ```bash
     touch .env
     ```

   - Add your API key to the `.env` file:

     ```env
     FMP_API_KEY=your_api_key_here
     ```

4. **Prepare Symbols File**

   - Create a text file (e.g., `symbols.txt`) containing the list of stock symbols you want to analyze, one per line.

     ```txt
     AAPL
     MSFT
     GOOG
     AMZN
     ```

## Usage

Run the script using the following command:

```bash
python analyst_prediction_analyzer.py --symbols symbols.txt
```

The utility will start running and perform the following tasks:

- **Weekly Analysis**: Every Monday at 9 AM, it will analyze analysts' predictions and update their accuracy data.
- **RSS Feed Updates**: Every 12 hours, it will generate or update the RSS feed (`top_analysts_feed.xml`) with the latest reports from the top analysts.

### Logs

- Activity is logged to `analyst_prediction_analyzer.log`. Check this file for detailed logs and error messages.

### RSS Feed

- The RSS feed is generated as `top_analysts_feed.xml` in the project directory.
- Users can subscribe to this feed using any RSS reader to receive updates on the top analysts.

## Configuration

- **Scheduling**: Adjust scheduling times by modifying the `schedule.every().monday.at("09:00").do(weekly_analysis)` and `schedule.every(12).hours.do(update_rss_feed)` lines in the script.
- **Minimum Predictions**: Change the `min_predictions` parameter in the `find_top_analysts` function to adjust the minimum number of predictions required for an analyst to be considered.

## How It Works

1. **Fetching Analyst Data**

   - Reads the list of stock symbols from the `symbols.txt` file.
   - Fetches the latest price targets for these symbols from the Financial Modeling Prep API.
   - Retrieves historical stock price data using `yfinance`.

2. **Analyzing Predictions**

   - Compares analysts' price targets to actual stock prices after a specified time horizon (default is 90 days).
   - Calculates the percentage error for each prediction.
   - Aggregates errors to compute an accuracy score for each analyst.
   - Updates analysts' accuracy data and saves it to `analyst_accuracy.csv`.

3. **Identifying Top Analysts**

   - Filters out analysts with fewer than the minimum required predictions.
   - Calculates a weighted score based on accuracy and the number of predictions.
   - Ranks analysts and selects the top 10.

4. **Generating RSS Feed**

   - Fetches the latest reports from the top analysts.
   - Generates an RSS feed (`top_analysts_feed.xml`) containing:
     - Analyst's name and company
     - Their accuracy and prediction volume
     - A brief description of their most recent reports or predictions

## Troubleshooting

- **API Errors**: If you encounter API rate limit errors, adjust the script to handle rate limiting or consider upgrading your API plan.
- **Missing Data**: Ensure that the symbols in your `symbols.txt` file are valid and that data is available for them.
- **Logging**: Refer to `analyst_prediction_analyzer.log` for detailed error messages and logs.
