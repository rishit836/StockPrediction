from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
model = load_model('models/ticker1')

def preprocess_and_predict(path):
    data = pd.read_csv(path)
    columns = data.columns
    columns = columns[1:]
    for i in columns:
        ms = MinMaxScaler()
        conversion =np.array( data[i]).reshape(-1, 1)
        data[i] = ms.fit_transform(conversion)
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
            return_dict[str(i)] = preds
            # if int(preds.max() - max_num()) > 0:
            #     return_dict[str(i) + "profit"] = "postive"
            # else:
            #     return_dict[str(i) + "profit"] = "negative"
                
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