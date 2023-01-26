# Stock Price Prediction using Polygon.io and LSTM

This script uses the [Polygon.io](https://polygon.io/) API to retrieve historical stock data and a Long Short-Term Memory (LSTM) network to predict the next day's closing price of a given stock. 
The script is written in Python, using libraries such as pandas, numpy, scikit-learn, keras and requests.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- keras
- requests

You can install the required packages by running:
pip install -r requirements.txt

Copy code

## Usage
1. Replace YOUR_API_KEY in the script with your own Polygon.io API key.
2. Run the script by providing the ticker symbol of the stock you want to predict as an argument. For example:
python stock_prediction.py TSLA

Copy code

## Output
The script will output the last closing price of the stock, the next day's predicted closing price, and the root mean squared error (RMSE) between the predictions and the actual values.

Additionally, it will also save a csv file with the historical data and the predicted closing prices, with the ticker symbol and the date it was created in the file name.

## Note
The script is trained on the last month of data. If the stock has not been publicly traded for a month or more, the script will not work as expected.

## Acknowledgements
This script is for educational purposes and not intended for financial advice. The accuracy of the predictions is not guaranteed and should not be used for investment decisions.