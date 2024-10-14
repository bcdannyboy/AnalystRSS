# AnalystRSS

**Analyze the analysts, then analyze their analysis**

AnalystRSS is a Python utility that fetches analyst price targets and stock prices to evaluate the prediction accuracy of financial analysts. It performs weekly analysis to identify the top analysts based on their historical performance and generates an RSS feed with the latest reports from these top analysts. The RSS feed is updated every 12 hours and served via an HTTP server.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Set Up Environment Variables](#set-up-environment-variables)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Configuration](#configuration)
- [Logging](#logging)
- [Scheduling Tasks](#scheduling-tasks)
- [API Key Acquisition](#api-key-acquisition)
- [Contributing](#contributing)

## Features

- **Analyst Accuracy Analysis**: Evaluates financial analysts by comparing their price targets to actual stock performance over a specified time horizon.
- **Top Analyst Identification**: Identifies the top analysts based on accuracy and prediction volume.
- **RSS Feed Generation**: Creates an RSS feed with the latest reports from the top analysts.
- **HTTP Server**: Serves the RSS feed via a built-in HTTP server.
- **Scheduled Tasks**: Automatically updates analyst accuracy data weekly and refreshes the RSS feed every 12 hours.
- **Multi-threaded Data Fetching**: Uses multi-threading to efficiently fetch data from APIs.
- **Rate Limiting**: Includes built-in rate limiting to comply with API usage policies.
- **Logging**: Provides detailed logs for monitoring and debugging purposes.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **pip** (Python package manager)

### Clone the Repository

```bash
git clone https://github.com/bcdannyboy/AnalystRSS.git
cd AnalystRSS
```

### Install Dependencies

It's recommended to use a virtual environment to avoid conflicts with other Python packages.

#### Using `venv`

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

#### Using `pipenv` (Optional)

If you prefer `pipenv`, you can use:

```bash
pip install pipenv
pipenv install
```

### Set Up Environment Variables

Create a `.env` file in the root directory of the project and add your [Financial Modeling Prep API](https://financialmodelingprep.com/) key:

```ini
FMP_API_KEY=your_api_key_here
```

## Usage

The script can be run from the command line with various options.

### Command-Line Arguments

- `--symbols`: **(Required)** Path to the text file containing stock symbols, one per line.
- `--ratelimit`: API rate limit (requests per minute). Default: `300`.
- `--timehorizon`: Time horizon in days to evaluate predictions. Default: `365`.
- `--host`: Host address to run the RSS server on. Default: `0.0.0.0`.
- `--port`: Port number to run the RSS server on. Default: `5000`.

### Examples

#### Basic Usage

```bash
python AnalystRSS.py --symbols symbols.txt
```

#### Custom Time Horizon and Rate Limit

```bash
python AnalystRSS.py --symbols symbols.txt --timehorizon 180 --ratelimit 200
```

#### Custom Host and Port for RSS Server

```bash
python AnalystRSS.py --symbols symbols.txt --host 127.0.0.1 --port 8080
```

## Configuration

- **Symbols File**: A text file containing the stock symbols you want to analyze, one symbol per line.
- **.env File**: Contains environment variables, specifically the `FMP_API_KEY`.
- **API Rate Limit**: Adjust the `--ratelimit` argument to match your API plan's limitations.
- **Time Horizon**: Set the `--timehorizon` argument to specify how many days ahead the analyst predictions should be evaluated.
- **Logging Level**: The logging level is set to `INFO` by default. To enable more detailed logs, change `level=logging.INFO` to `level=logging.DEBUG` in the script.

## Logging

Logs are written to both the console and a file named `analyst_prediction_analyzer.log`. The log includes timestamps, log levels, and messages to help you monitor the script's execution and troubleshoot if necessary.

## Scheduling Tasks

The script uses the `schedule` library to automate tasks:

- **Weekly Analysis**: Runs every Sunday at 12:00 AM to update analyst accuracy data.
- **RSS Feed Update**: Runs every 12 hours to refresh the RSS feed with the latest reports.

These schedules are defined in the `main()` function:

```python
# Schedule weekly_analysis() every Sunday at 12 AM
schedule.every().sunday.at("00:00").do(weekly_analysis)

# Schedule update_rss_feed() every 12 hours
schedule.every(12).hours.do(update_rss_feed)
```

## API Key Acquisition

The script requires an API key from [Financial Modeling Prep](https://financialmodelingprep.com/).

1. Sign up for a free account at [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs).
2. Obtain your API key from the dashboard.
3. Add the API key to your `.env` file:

```ini
FMP_API_KEY=your_api_key_here
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -am 'Add your feature'`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Submit a pull request.
