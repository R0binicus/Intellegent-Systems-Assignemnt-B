# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, InputLayer, SimpleRNN, GRU
from keras.callbacks import EarlyStopping

import matplotlib
import os
from os.path import exists
import time
from collections import deque
from datetime import timedelta, date
from datetime import datetime
import random
from parameters import *
import mplfinance as mpl 
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt



# Temporary
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import load_model


# Train and test data global variables for setting
trainData = None
testData = None
fullData = None

# Function for checking if the data is already downloaded (needs internet connection to download)
# If the data is NOT in a file, it downloads the data and makes a csv file and returns the data
# If the data IS in a file, it reads the data from the file and returns the data
def checkFiles(filename):
    if (os.path.exists(filename)):
        #Read csv file and return the data inside
        data = pd.read_csv(filename)
        # NaN values from pandas are values that are not present. For example in stocks if the stock data for a 
        # specific day was not recorded, i believe it would still have a record for that day, only the values 
        # would be NaN or 'Not a Number'
        
        #what dropna() does is simply remove the missing values from the dataset
        data.dropna(inplace=True)
        return data
    else:
        #Download data from online
        data = yf.download(COMPANY, start=TRAIN_START, end=TEST_END, progress=False)

        # Save data to csv   file
        data.to_csv(filename)
        # For some reason it needs to read it from the file otherwise it won't work
        data = pd.read_csv(filename)
        
        if (STOREFILE == False):
            os.remove(filename) #Remove stored file 
        # remove NaN values from the dataset
        data.dropna(inplace=True)
        return data

 #Base function for future purposes
#def getData(filename):
#    df = checkFiles(filename)
#
#    df['Date'] = pd.to_datetime(df['Date'])
#    df = df.set_index(df['Date'])
#    df = df.sort_index()
#
#    # create train test partition
#    global trainData
#    #trainData = df
#    trainData = df[TRAIN_START:TRAIN_END]
#    global testData
#    testData = df[TEST_START:TEST_END]
#    #testData = df
#    print('Train Dataset:',trainData.shape)
#    print('Test Dataset:',testData.shape)
#
#    #trainData.to_csv("trainfilename.csv")
#    #testData.to_csv("testfilename.csv")        #test that the data is split correctly

# This function gets the datafile name as well as the split date
# it then runs the file checker to get the dataset, then splits the dataset at the split date
def getDataSplitDate(filename, splitDate):
    df = checkFiles(filename)

    # Make it know that the date colunm is indeed a date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    #Convert input to datetime, add 1 day, then convert back to string
    date = datetime.strptime(splitDate, '%Y-%m-%d')
    testStartDate = date + timedelta(days=1)
    testStartDate = testStartDate.strftime('%Y-%m-%d')

    # Code for testing dates
    print('splitDate:',splitDate)
    print('testStartDate:',testStartDate)

    splitDate = '2020-09-01'
    testStartDate = '2020-09-02'

    # Code for testing dates
    print('splitDate:',splitDate)
    print('testStartDate:',testStartDate)

    fullData = df

    # create train/test partition
    global trainData
    trainData = df[TRAIN_START:splitDate]
    trainData = trainData.drop(trainData.columns[[0]], axis=1)
    global testData
    testData = df[testStartDate:TEST_END]
    testData = testData.drop(testData.columns[[0]], axis=1)
    print('Train Dataset:',trainData.shape)
    print('Test Dataset:',testData.shape)

    #trainData.to_csv("trainfilename.csv")
    #testData.to_csv("testfilename.csv")        #test that the data is split correctly


# This function gets the datafile name as well as the 'ratio' number
# it then runs the file checker to get the dataset, then splits the dataset at the split date
def getDataRatio(filename, ratio):
    df = checkFiles(filename)

    # Make it know that the date colunm is indeed a date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(df['Date'])
    df = df.sort_index()

    # Convert strings to dates
    date1 = datetime.strptime(TRAIN_START, '%Y-%m-%d')
    date2 = datetime.strptime(TEST_END, '%Y-%m-%d')

    # do math to get the date we want
    trainEndDate = date2 + (date1 - date2) / ratio

    #Convert input to datetime, add 1 day
    print("Middle : " + trainEndDate.strftime('%Y-%m-%d'))
    testStartDate = trainEndDate + timedelta(days=1)
    # Convert back to string
    testStartDate = testStartDate.strftime('%Y-%m-%d')
    trainEndDate = trainEndDate.strftime('%Y-%m-%d') 

    # Code for testing dates
    print('trainEndDate Dataset:',trainEndDate)
    print('testStartDate:',testStartDate)

    fullData = df

    # create train/test partition
    global trainData
    trainData = df[TRAIN_START:trainEndDate]
    trainData = trainData.drop(trainData.columns[[0]], axis=1)
    global testData
    testData = df[testStartDate:TEST_END]
    testData = testData.drop(testData.columns[[0]], axis=1)

    #trainData.to_csv("trainfilename.csv")
    #testData.to_csv("testfilename.csv")        #test that the data is split correctly

    # link: https://www.relataly.com/stock-market-prediction-using-multivariate-time-series-in-python/1815/

def ARIMA_prediction():

    #series = trainData
    # split into train and test sets
    #X = series['Close'].values
    #size = int(len(X) * 0.66)
    #train, test = X[0:size], X[size:len(X)]
    train = trainData['Close'].values

    test1 = testData[1:]
    test = test1['Close'].values

    
    
    history = [x for x in train]
    predictions = list()
    my_seasonal_order = (1, 1, 0, 6)

    # walk-forward validation
    for t in range(len(test)):
        if SARIMA:
            model = ARIMA(history, order=(5,1,0), seasonal_order=my_seasonal_order)
        else:
            model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    return predictions


    #LINK https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/


    #LINK https://medium.com/@cortexmoldova/arima-time-series-forecasting-model-with-python-5b0cfdbb08fa
    # Maybe use this one

def multivariate_prediction(layer_num, layer_size, layer_name):
    PREDICT_COLUNM = "Close"
    FEATURE_COLUNMS = ['Open','High','Low','Close','Adj Close','Volume']

    # make a copy of the train and test dataframes
    train_df = trainData.sort_values(by=['Date']).copy()
    test_df = testData.sort_values(by=['Date']).copy()
    # Add dummy column and set dummy values for sacling in the future
    train_df_ext = train_df.copy()
    train_df_ext['Dummy'] = train_df_ext['Close']
    test_df_ext = test_df.copy()
    test_df_ext['Dummy'] = test_df_ext['Close']
    # Get the number of rows in the data
    nrows = train_df.shape[0]
    # Convert the data to numpy values
    np_train_unscaled = np.array(train_df)
    np_test_unscaled = np.array(test_df)
    np_data = np.reshape(np_train_unscaled, (nrows, -1))
    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_train_scaled = scaler.fit_transform(np_train_unscaled)
    np_test_scaled = scaler.fit_transform(np_test_unscaled)
    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(train_df_ext['Close'])
    df_Close2 = pd.DataFrame(test_df_ext['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
    np_Close_scaled2 = scaler_pred.fit_transform(df_Close2)
    # Set Prediction Index
    index_Close = train_df.columns.get_loc("Close")
    # Create the training and test data
    train_data = np_train_scaled
    test_data = np_test_scaled
    # Here, we create N samples, LOOKBACK_DAYS time steps per sample, and 6 features
    def partition_dataset(LOOKBACK_DAYS, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(LOOKBACK_DAYS, data_len):
            x.append(data[i-LOOKBACK_DAYS:i,:]) #contains LOOKBACK_DAYS values 0-LOOKBACK_DAYS * colunms
            y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
        # Convert x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y
    # Generate training data and test data
    x_train, y_train = partition_dataset(LOOKBACK_DAYS, train_data)
    x_test, y_test = partition_dataset(LOOKBACK_DAYS, test_data)

    # Configure the neural network model
    model = Sequential()

    #Add layers to network using for each loop, which takes the layer_num to determine how many layers are added
    for i in range(layer_num):
        if i == 0:
            # first layer
            model.add(layer_name(layer_size, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        elif i == layer_num - 1:
            # last layer
            model.add(layer_name(layer_size, return_sequences=False))
        else:
            # hidden layers
            model.add(layer_name(layer_size, return_sequences=True))
    
    # Prediction of the next closing value of the stock price
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Training the model
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    # Get the predicted values
    y_pred_scaled = model.predict(x_test)
    # Unscale the predicted values
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    
    return y_pred

def createModel(layer_num, layer_size, layer_name, dropout):
    #Declare some variables so the model knows whats what
    PRICE_VALUE = "Close"


    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(trainData[PRICE_VALUE].values.reshape(-1, 1)) 

    # To store the training data
    x_train = []
    y_train = []

    scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
    # Prepare the data
    for x in range(LOOKBACK_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x-LOOKBACK_DAYS:x])
        y_train.append(scaled_data[x])

    # Convert them into an array
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Now, x_train is a 2D array(p,q) where p = len(scaled_data) - LOOKBACK_DAYS
    # and q = LOOKBACK_DAYS; while y_train is a 1D array(p)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
    # is an array of p inputs with each input being a 2D array

    model = Sequential() # Basic neural network
    #Add layers to network using for each loop, which takes the layer_num to determine how many layers are added
    for i in range(layer_num):
        if i == 0:
            # first layer
            model.add(layer_name(layer_size, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        elif i == layer_num - 1:
            # last layer
            model.add(layer_name(layer_size, return_sequences=False))
        else:
            # hidden layers
            model.add(layer_name(layer_size, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    
    # Prediction of the next closing value of the stock price
    model.add(Dense(units=1)) 
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Now we are going to train this model with our training data 
    # (x_train, y_train)
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Return completed model to be tested 
    return model

def runTest():
    arima_pred = ARIMA_prediction()
    #model2 = model1
    if MULTIVARIATE:
        multi_pred = multivariate_prediction(LAYER_NUM, LAYER_SIZE, LAYER_NAME)
    
    #createModel2(layer_num, layer_size, layer_name, dropout):
    model = createModel(LAYER_NUM, LAYER_SIZE, LAYER_NAME, DROPOUT)

    #Make sure it knows that testData is refering to the global
    global testData

    PRICE_VALUE = "Close"

    PREDICT_COLUNM = "Close"

    scaler = MinMaxScaler(feature_range=(0, 1)) 

    scaled_data = scaler.fit_transform(trainData[PRICE_VALUE].values.reshape(-1, 1)) 

    

    #------------------------------------------------------------------------------
    # Test the model accuracy on existing data
    #------------------------------------------------------------------------------

    # The above bug is the reason for the following line of code
    testData = testData[1:]

    testData.to_csv("testfilename.csv") 

    actual_prices = testData[PRICE_VALUE].values

    total_dataset = pd.concat((trainData[PRICE_VALUE], testData[PRICE_VALUE]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(testData) - LOOKBACK_DAYS:].values
    # We need to do the above because to predict the closing price of the fisrt
    # LOOKBACK_DAYS of the test period [TEST_START, TEST_END], we'll need the 
    # data from the training period

    model_inputs = model_inputs.reshape(-1, 1)
    # TO DO: Explain the above line

    model_inputs = scaler.transform(model_inputs)
    # We again normalize our closing price data to fit them into the range (0,1)
    # using the same scaler used above 
    # However, there may be a problem: scaler was computed on the basis of
    # the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
    # but there may be a lower/higher price during the test period 
    # [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
    # greater than one)
    # We'll call this ISSUE #2

    # TO DO: Generally, there is a better way to process the data so that we 
    # can use part of it for training and the rest for testing. You need to 
    # implement such a way

    #------------------------------------------------------------------------------
    # Make predictions on test data
    #------------------------------------------------------------------------------
    x_test = []
    for x in range(LOOKBACK_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - LOOKBACK_DAYS:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # TO DO: Explain the above 5 lines

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    # Clearly, as we transform our data into the normalized range (0,1),
    # we now need to reverse this transformation 

    #------------------------------------------------------------------------------
    # Predict next day
    #------------------------------------------------------------------------------

    futurePrice = []


    i = 0
 
    # for each day in the future to predict
    while i < PREDICTION_DAYS:
        i += 1

        # make it so it does (or redoes) the declaring of real_data based on the last PREDICTION_DAYS amount of days
        real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
 
        # make prediction of next day
        prediction = model.predict(real_data)

        # unscale and flatten prediction data
        scaledPrdiction = scaler.inverse_transform(prediction)

        # add predicted data to future price
        futurePrice.append(scaledPrdiction.flatten()[0])

        # set prediction to be the flattened but still scaled data
        prediction = prediction.flatten()[0]

        print(futurePrice)

        # set model inputs to be dataframe, add predicted data to it, then make it a numpy array again
        model_inputs = pd.DataFrame(model_inputs)
        model_inputs.loc['0'] = prediction
        model_inputs = model_inputs.to_numpy()

    # Make it so the future predicted days appear after the test data
    df_futurePrices = pd.DataFrame(columns=['Index','Forecast'])
    DF = pd.DataFrame(predicted_prices)
    df_futurePrices['Index'] = range(DF.index[-1] + 1, DF.index[-1] + 1 + PREDICTION_DAYS)
    df_futurePrices = df_futurePrices.set_index("Index")
    df_futurePrices['Forecast'] = np.array(futurePrice)

    # Add LOOKBACK_DAYS to the start of the multivatiate so it's start on the chart is delayed
    if MULTIVARIATE:
        df_multiPrices = pd.DataFrame(columns=['Index','Close'])
        DF2 = pd.DataFrame(multi_pred)
        df_multiPrices['Index'] = range(LOOKBACK_DAYS, DF2.index[-1] + 1 + LOOKBACK_DAYS)
        df_multiPrices = df_multiPrices.set_index("Index")
        df_multiPrices['Close'] = np.array(multi_pred)

    ensemble_preds = None
    if ENSEMBLE:
        predicted_prices_list = predicted_prices.tolist()

        i = 0
        
        for value in arima_pred:
            ensemble_preds = np.append(ensemble_preds, (arima_pred[i] + predicted_prices_list[i])/2)
            i += 1


    # A few concluding remarks here:
    # 1. The predictor is quite bad, especially if you look at the next day 
    # prediction, it missed the actual price by about 10%-13%
    # Can you find the reason?
    # 2. The code base at
    # https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
    # gives a much better prediction. Even though on the surface, it didn't seem 
    # to be a big difference (both use Stacked LSTM)
    # Again, can you explain it?
    # A more advanced and quite different technique use CNN to analyse the images
    # of the stock price changes to detect some patterns with the trend of
    # the stock price:
    # https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
    # Can you combine these different techniques for a better prediction??

    #------------------------------------------------------------------------------
    # Plot the test predictions
    ## To do:
    # 1) Candle stick charts
    # 2) Chart showing High & Lows of the day
    # 3) Show chart of next few days (predicted)
    #------------------------------------------------------------------------------

    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    if ENSEMBLE:
        plt.plot(ensemble_preds, color="green", label=f"Ensemble Predicted {COMPANY} Price")
    else:
        plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
        plt.plot(df_futurePrices, color="orange", label=f"Predicted {COMPANY} Future Price")
        if MULTIVARIATE: # Display Multivariate data if it is enabled
            plt.plot(df_multiPrices, color="red", label=f"Predicted {COMPANY} multivariate Price")
    
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.show()

    #candlestickChart()


def candlestickChart(filename):
    # Get data
    df = checkFiles(filename)

    # Make it know that the date colunm is indeed a date
    df['Date'] = pd.to_datetime(df['Date'])
    # Set the index of the dataframe to be the date colunm
    df = df.set_index(df['Date'])
    df = df.sort_index()

    # Get the last n days 
    actual_prices_small = df[-NDAYS:]

    # Plot the candlestick chart
    mpl.plot(actual_prices_small.set_index("Date"), type="candle", style="charles", title='Candlestick Chart')
    # Uses mplfinance  to make the candlestick chart
    # uses the date colunm as index
    # uses the 'charles' stle to make the decreasing days red and increasing days green

def boxplotChart(filename):
    # Get data
    df = checkFiles(filename)
    # Get the last n days 
    actual_prices_small = df[-NDAYS:]

    # Plot the boxplot chart
    fig = actual_prices_small[['Open', 'Close', 'Low', 'High']].plot(kind='box', title='Boxplot Chart', grid=True)
    # Uses Matplotlib  to make the boxplot chart
    # uses grid=True to make a grid so the chart is easier to read

    # Add axis labels
    plt.xlabel("Stock Data Type")
    plt.ylabel("Price")

    # Display the plot
    plt.show()

def Main(): #Main function for deciding which split method was chosen
    #Make filename for the saved data file
    ticker_data_filename = os.path.join("data", f"{COMPANY}_{TRAIN_START}_{TEST_END}.csv")
    
    # Switch for checking which mode to run the program in
    match MODE: 
        case 1: #Predict with date split
            getDataSplitDate(ticker_data_filename, SPLIT_DATE)
            runTest()

        case 2: #Predict with ratio split
            getDataRatio(ticker_data_filename, RATIO)
            runTest()

        case 3: #Candlestick Chart
            candlestickChart(ticker_data_filename)

        case 4: #Boxplot Chart
            boxplotChart(ticker_data_filename)

        case _: #Predict with random date split
            #Convert dates to datetime
            dateStart = datetime.strptime(TRAIN_START, '%Y-%m-%d')
            dateEnd = datetime.strptime(TEST_END, '%Y-%m-%d')
            #Get random date inbetween the start and end
            random_date = dateStart + (dateEnd - dateStart) * random.random()
            #convert back to string
            random_date = random_date.strftime('%Y-%m-%d')
            #Use random date as split
            getDataSplitDate(ticker_data_filename, random_date)
            runTest()

Main()
    