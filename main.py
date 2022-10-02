from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
model = load_model('models/ticker1')

def preprocess_and_predict(path):
    data = pd.read_csv(path)
    columns = data.columns
    columns = columns[1:]
    df = data.T
    df.columns = data.ticker.to_list()
    df = df.iloc[1:, :]
    days_lst = df.index.to_list()
    return_dict = {}
    temp_lst = []
    for i in range(0, 84):
        for j in days_lst:
            temp_lst.append(df.at[j,i])
            out_df = pd.DataFrame(temp_lst)
            max_num = out_df.max()
            preds = model.predict(out_df)
            if preds.max() > preds.mean():
                print('profit')
                return_dict[str(i)] = "positive"
            elif ((preds.mean() - preds.max()) < .02): # Risk Factor
                print("profit")
                return_dict[str(i)] = "positive"
            else:
                print("loss")
                return_dict[str(i)] = "negative"

                
        temp_lst = []

    return return_dict
            

def make_trades():
    path = input("Enter the path of the dataset")
    if path != '':
        preprocess_and_predict(path)
    elif path == '':
        print("Path cannot be empty")

if __name__ == '__main__':
    print(make_trades())