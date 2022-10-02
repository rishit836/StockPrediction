from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

model = load_model('models/ticker1')


def preprocess_and_predict(path):
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
    return_dict = {}
    temp_lst = []
    for i in range(0, len(ticker_lst)):
        for j in days_lst:
            temp_lst.append(df.at[j, i])
            out_df = pd.DataFrame(temp_lst)
        preds = model.predict(out_df)
        if preds.mean() > preds.max():
            return_dict[str(i)] = "positive"

        if ((round(preds.max(), 2) - round(preds.mean(), 2)) > .01):  # Risk Factor
            risk_fac = round(preds.max(), 2) - round(preds.mean(), 2)
            return_dict[str(i)] = "positive"

        else:
            print("loss")
            return_dict[str(i)] = "negative"

        temp_lst = []

    return return_dict


def make_trades():
    path = input("Enter the path of the dataset")
    if path != '':
        print(preprocess_and_predict(path))
    elif path == '':
        print("Path cannot be empty")


if __name__ == '__main__':
    make_trades()
