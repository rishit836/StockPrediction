from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

model = load_model('models/ticker1')


def maxProfit(price, start, end):
    # If the stocks can't be bought
    if (end <= start):
        return 0;

    # Initialise the profit
    profit = 0;

    # The day at which the stock
    # must be bought
    for i in range(start, end, 1):

        # The day at which the
        # stock must be sold
        for j in range(i + 1, end + 1):

            # If buying the stock at ith day and
            # selling it at jth day is profitable
            if (price[j] > price[i]):
                # Update the current profit
                curr_profit = price[j] - price[i] + maxProfit(price, start, i - 1) + maxProfit(price, j + 1, end);

                # Update the maximum profit so far
                profit = max(profit, curr_profit);

    return profit;


def stockBuySell(price, n):
    # Prices must be given for at
    # least two days
    if (n == 1):
        return

    # Traverse through given price array
    i = 0
    while (i < (n - 1)):

        # Find Local Minima
        # Note that the limit is (n-2) as
        # we are comparing present element
        # to the next element
        while ((i < (n - 1)) and (price[i + 1] <= price[i])):
            i += 1

        # If we reached the end, break
        # as no further solution possible
        if (i == n - 1):
            break

        # Store the index of minima
        buy = i
        i += 1

        # Find Local Maxima
        # Note that the limit is (n-1) as we are
        # comparing to previous element
        while ((i < n) and (price[i] >= price[i - 1])):
            i += 1
        # Store the index of maxima
        sell = i - 1

        return buy, sell


def preprocess_and_predict(path, holdings):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("File Not Found Please retry")
        raise FileNotFoundError

    columns = data.columns
    columns = columns[1:]
    for i in columns:
        ms = MinMaxScaler()
        conversion = np.array(data[i]).reshape(-1, 1)
        data[i] = ms.fit_transform(conversion)
    df = data.T
    ticker_lst = data.ticker.to_list()
    df.columns = data.ticker.to_list()
    df = df.iloc[1:, :]
    days_lst = df.index.to_list()
    temp_lst = []
    profits = []
    initial_prices = []
    buy_sell = []

    for i in range(0, 5):
        for j in days_lst:
            if j == 'day_0':
                initial_price = df.at[j, i]
                initial_prices.append(initial_price)  # getting the price of stock at day_0 to get the initial price
            temp_lst.append(df.at[j, i])
            out_df = pd.DataFrame(temp_lst)
            preds = model.predict(out_df)
            preds = list(preds)
            preds.sort()
            max_profit = maxProfit(preds, 0, len(preds) - 1)  # to get the price of the stock
            buy, sell = stockBuySell(preds, len(preds))
            buy_sell.append([buy, sell])
            profits.append(max_profit)
            print("Done")

            temp_lst = []

            return profits, 5, initial_prices, buy_sell


# DF variable expects a path for dataframe

def make_trades(df, holdings, current_money, day):
    path = df
    if path != '' and day == 0:
        profits, ticker_num, initial_prices, buy_sell = preprocess_and_predict(path,
                                                                               holdings)  # profits for all the stocks ticker-wise
        with open('later_var.txt', 'w') as f:
            # just to get the predictions one time to less the runtime
            f.write(str(profits))
            f.write('\n')
            f.write(str(ticker_num))
            f.write('\n')
            f.write(str(initial_prices))
            f.write('\n')
            f.write(str(buy_sell))
    elif path == '':
        print("Path cannot be empty")

    with open('later_var.txt', "r") as f1:
        data_read = f1.readlines()
        for i in range(0, len(data_read)):
            data_read[i] = data_read[i].rstrip()

        for x in range(0, len(data_read)):
            if x == 0:
                profits = list(data_read[x])
            if x == 1:
                ticker_num = int(data_read[x])
            if x == 2:
                initial_prices = list(data_read[x])
            if x == 3:
                buy_sell = list(data_read[x])

    for i in range(0, day):
        stocks = profits
        stocks.sort(reverse=True)
        return_dict = {}
        for ticker_most_profitable in range(0, ticker_num):
            value = stocks[ticker_most_profitable]
            profit_ticker = profits.index(value)
            for j in range(0, len(buy_sell)):
                if j == profit_ticker:
                    print(j)
                    if day == buy_sell[j][0]:
                        print("Buy The Stock at day", day)
                        current_money -= profits[j] * 50  # by default the bot's gonna buy 50 stocks
                        return_dict[str(j)] = '+50'
                    if day == buy_sell[j][1]:
                        print('sell the stock at day', day)
                        current_money += profits[j] * 50  # by default the bot's gonna sell 50 stocks
                        return_dict[str(j)] = '-50'

                    return return_dict
