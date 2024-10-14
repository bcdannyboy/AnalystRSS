# AnalystRSS

**Analyze the analysts, then analyze their analysis**

---

## Introduction

AnalystRSS is a powerful utility designed to evaluate the prediction accuracy of financial analysts. By fetching analysts' price targets and comparing them with actual stock prices over a customizable time horizon, it identifies the top-performing analysts based on historical performance. The tool then generates an RSS feed with the latest reports from these top analysts, updating it every 12 hours. AnalystRSS automates the entire process, providing investors and researchers with insights into analyst performance to inform investment decisions.

---

## Features

- **Customizable Time Horizon**: Specify the number of days over which to evaluate analyst predictions.
- **Analyst Performance Evaluation**: Fetches price targets and compares them to actual stock prices to assess accuracy.
- **Top Analysts Identification**: Ranks analysts based on accuracy and prediction volume, selecting the top 10.
- **RSS Feed Generation**: Generates and updates an RSS feed with the latest reports from top analysts every 12 hours.
- **Automated Scheduling**: Schedules weekly analyses and RSS feed updates.
- **Multithreaded Data Fetching**: Utilizes multithreading for efficient data retrieval.
- **API Rate Limiting**: Implements rate limiting to comply with API usage policies.
- **Detailed Logging**: Provides comprehensive logging for monitoring and debugging.

---

## Installation

### Prerequisites

- **Python3**
- **Financial Modeling Prep API Key**

### Clone the Repository

```bash
git clone https://github.com/yourusername/AnalystRSS.git
cd AnalystRSS
```

### Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the project root directory and add your Financial Modeling Prep API key:

```
FMP_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual API key.

### Prepare the Symbols File

Create a `symbols.txt` file in the project directory with a list of stock symbols to analyze, one per line:

```
AAPL
MSFT
GOOGL
```

A comprehensive list of stock symbols is provided in `symbols.txt`.

---

## Usage

Run the script using the following command:

```bash
python AnalystRSS.py --symbols symbols.txt [--timehorizon TIME_HORIZON] [--ratelimit RATE_LIMIT]
```

### Command-Line Arguments

- `--symbols`: **(Required)** Path to the symbols text file.
- `--timehorizon`: Time horizon in days to evaluate predictions. Default is **365** days.
- `--ratelimit`: API rate limit (requests per minute). Default is **300**.

### Examples

- Run with default settings:

  ```bash
  python AnalystRSS.py --symbols symbols.txt
  ```

- Specify a custom time horizon and rate limit:

  ```bash
  python AnalystRSS.py --symbols symbols.txt --timehorizon 180 --ratelimit 200
  ```

---

## Configuration

### Adjusting the Time Horizon

The `--timehorizon` argument allows you to specify the number of days over which analyst predictions are evaluated. For example, `--timehorizon 180` evaluates predictions over a 6-month period.

### API Rate Limiting

To comply with API usage policies, the script implements rate limiting. You can adjust the rate limit using the `--ratelimit` argument.

---

## Scheduling

The script uses the `schedule` library to automate tasks:

- **Weekly Analysis**: The `weekly_analysis` function is scheduled to run every Sunday at Midnight. It fetches the latest price targets, compares them to actual stock prices, updates analysts' accuracy data, and identifies the top analysts.

- **RSS Feed Updates**: The `update_rss_feed` function is scheduled to run every 12 hours. It generates or updates the RSS feed with the latest reports from the top analysts.

### Modifying the Schedule

You can adjust the scheduling by modifying the `main()` function in `AnalystRSS.py`:

```python
# Schedule weekly_analysis() every Sunday at Midnight
schedule.every().Sunday.at("00:00").do(weekly_analysis)

# Schedule update_rss_feed() every 12 hours
schedule.every(12).hours.do(update_rss_feed)
```

---

## Output

### Generated Files

- **analyst_accuracy.csv**: Contains the calculated skill scores and accuracy metrics for analysts.
- **top_analysts_feed.xml**: The generated RSS feed with the latest reports from the top analysts.
- **analyst_prediction_analyzer.log**: Log file containing detailed execution logs.

---

## How It Works

1. **Fetch Price Targets**: The script fetches price targets for the specified symbols from the Financial Modeling Prep API.

2. **Fetch Historical Data**: It retrieves historical stock prices using the yfinance library.

3. **Analyze Analyst Skill**: The script calculates the prediction errors for each analyst over the specified time horizon and computes a skill score based on accuracy and breadth.

4. **Identify Top Analysts**: Analysts are ranked based on their skill score, and the top 10 are selected.

5. **Generate RSS Feed**: An RSS feed is generated or updated with the latest reports from the top analysts.

6. **Scheduling**: The entire process is automated using scheduled tasks.

---

## Logging

The script provides detailed logging to both the console and a log file (`analyst_prediction_analyzer.log`). This includes information on data fetching, analysis progress, and any errors encountered.

You can adjust the logging level by modifying the logging configuration in `AnalystRSS.py`:

```python
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more detailed logs
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("analyst_prediction_analyzer.log"),
        logging.StreamHandler()
    ]
)
```

---

## Dependencies

All required dependencies are listed in `requirements.txt`:

```plaintext
requests
pandas
yfinance
schedule
feedgen
python-dotenv
tqdm
python-dateutil
```

Install them using:

```bash
pip install -r requirements.txt
```
