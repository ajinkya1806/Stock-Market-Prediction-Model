
# Stock Market Prediction Project

## Description
This project focuses on predicting stock market trends using machine learning models. It includes data preprocessing, model training, and prediction functionalities. The project is implemented in Python and leverages Jupyter Notebooks for experimentation and visualization.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data](#data)
- [Models](#models)


## Project Structure
```
app.py                        # Main application script
data/                         # Directory containing stock market data in CSV format
models/                       # Pre-trained machine learning models
Jupyter Notebook files/       # Notebooks for training and prediction
requirements.txt              # Python dependencies
README.md                     # Project documentation
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the main application**:
   Use `app.py` to load models and make predictions.
   ```bash
   python app.py
   ```

2. **Train models**:
   Open the Jupyter Notebook [`Jupyter Notebook files/Model_Training.ipynb`](Jupyter%20Notebook%20files/Model_Training.ipynb) to train new models using the provided data.

3. **Make predictions**:
   Use the notebook [`Jupyter Notebook files/stock_market_prediction_model.ipynb`](Jupyter%20Notebook%20files/stock_market_prediction_model.ipynb) to make predictions on stock market trends.

## Features
- Pre-trained models for stock market prediction.
- Jupyter Notebooks for model training and prediction.
- CSV data files for multiple companies.
- Modular structure for easy extension.

## Data
The `data/` directory contains stock market data in CSV format for the following companies:
- Bharti Airtel (`BHARTIARTL.csv`)
- HDFC Bank (`HDFCBANK.csv`)
- ITC (`ITC.csv`)
- Reliance (`RELIANCE.csv`)
- TCS (`TCS.csv`)

## Models
The `models/` directory contains pre-trained models:
- `Airtel_model.h5`
- `HDFC_model.h5`
- `RELIANCE_model.h5`
- `TCS_model.h5`
- `rnn_model.h5` (general RNN model)



