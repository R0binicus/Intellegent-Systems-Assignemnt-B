import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
import numpy as np
from datetime import datetime
import calendar
from datetime import timedelta
import datetime as dt
def add_month(df, forecast_length, forecast_period):
    end_point = len(df)
    df1 = pd.DataFrame(index=range(forecast_length), columns=range(2))
    df1.columns = ['SaleQty', 'date']
    df = df.append(df1)
    df = df.reset_index(drop=True)
    x = df.at[end_point - 1, 'date']
    x = pd.to_datetime(x, format='%Y-%m-%d')
    days_in_month=calendar.monthrange(x.year, x.month)[1]
    if forecast_period == 'Week':
        for i in range(forecast_length):
            df.at[df.index[end_point + i], 'date'] = x + timedelta(days=7 + 7 * i)
            df.at[df.index[end_point + i], 'SaleQty'] = 0
    elif forecast_period == 'Month':
        for i in range(forecast_length):
            df.at[df.index[end_point + i], 'date'] = x + timedelta(days=days_in_month + days_in_month * i)
            df.at[df.index[end_point + i], 'SaleQty'] = 0
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['month'] = df['date'].dt.month
    df = df.drop(['date'], axis=1)
    return df
def create_lag(df3):
    dataframe = DataFrame()
    for i in range(12, 0, -1):
        dataframe['t-' + str(i)] = df3.SaleQty.shift(i)
    df4 = pd.concat([df3, dataframe], axis=1)
    df4.dropna(inplace=True)
    return df4
def randomForest(df1, forecast_length, forecast_period):
    df3 = df1[['SaleQty', 'date']]
    df3 = add_month(df3, forecast_length, forecast_period)
    finaldf = create_lag(df3)
    finaldf = finaldf.reset_index(drop=True)
    n = forecast_length
    end_point = len(finaldf)
    x = end_point - n
    finaldf_train = finaldf.loc[:x - 1, :]
    finaldf_train_x = finaldf_train.loc[:, finaldf_train.columns != 'SaleQty']
    finaldf_train_y = finaldf_train['SaleQty']
    print("Starting model train..")
    rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=1), 4)
    fit = rfe.fit(finaldf_train_x, finaldf_train_y)
    print("Model train completed..")
    print("Creating forecasted set..")
    yhat = []
    end_point = len(finaldf)
    n = forecast_length
    df3_end = len(df3)
    for i in range(n, 0, -1):
        y = end_point - i
        inputfile = finaldf.loc[y:end_point, :]
        inputfile_x = inputfile.loc[:, inputfile.columns != 'SaleQty']
        pred_set = inputfile_x.head(1)
        pred = fit.predict(pred_set)
        df3.at[df3.index[df3_end - i], 'SaleQty'] = pred[0]
        finaldf = create_lag(df3)
        finaldf = finaldf.reset_index(drop=True)
        yhat.append(pred)
    yhat = np.array(yhat)
    print("Forecast complete..")
    return yhat
predicted_value=randomForest(jeans_data, 6, 'Month')