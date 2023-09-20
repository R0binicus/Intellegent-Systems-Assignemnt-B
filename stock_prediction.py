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

import math
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
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, InputLayer, SimpleRNN, GRU
from keras.callbacks import EarlyStopping



# To Remove

import seaborn as sns # Visualization
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})


# new

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

    # construct the model
    # model = create_model(N_STEPS, len(FEATURE_COLUMNS), units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, 
    #                 loss=LOSS,  optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    #(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
    #            loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):

    #(LOOKBACK_DAYS, n_features, units=50, cell=LSTM, n_layers=2, dropout=0.2,
    #            loss="mean_squared_error", optimizer="adam", bidirectional=False):


    # link: https://www.relataly.com/stock-market-prediction-using-multivariate-time-series-in-python/1815/

def multivariate_prediction(layer_num, layer_size, layer_name):
    PREDICT_COLUNM = "Close"
    FEATURE_COLUNMS = ['Open','High','Low','Close','Adj Close','Volume']

    train_df = trainData.sort_values(by=['Date']).copy()
    test_df = testData.sort_values(by=['Date']).copy()
    data_filtered = train_df
    data_filtered2 = test_df
    # We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['Close']
    data_filtered_ext2 = data_filtered2.copy()
    data_filtered_ext2['Prediction'] = data_filtered_ext2['Close']
    # Get the number of rows in the data
    nrows = data_filtered.shape[0]
    # Convert the data to numpy values
    np_train_unscaled = np.array(data_filtered)
    np_test_unscaled = np.array(data_filtered2)
    np_data = np.reshape(np_train_unscaled, (nrows, -1))
    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_train_scaled = scaler.fit_transform(np_train_unscaled)
    np_test_scaled = scaler.fit_transform(np_test_unscaled)
    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(data_filtered_ext['Close'])
    df_Close2 = pd.DataFrame(data_filtered_ext2['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
    np_Close_scaled2 = scaler_pred.fit_transform(df_Close2)
    # Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = 50
    # Prediction Index
    index_Close = train_df.columns.get_loc("Close")
    # Create the training and test data
    train_data = np_train_scaled
    test_data = np_test_scaled
    # The RNN needs data with the format of [samples, time steps, features]
    # Here, we create N samples, sequence_length time steps per sample, and 6 features
    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
            y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y
    # Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)

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
    history = model.fit(x_train, y_train, 
                        batch_size=32, 
                        epochs=10,
                        validation_data=(x_test, y_test)
                       )
    # Get the predicted values
    y_pred_scaled = model.predict(x_test)
    # Unscale the predicted values
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    
    return y_pred


# The date from which on the date is displayed
    #display_start_date = "2019-01-01" 
    # Add the difference between the valid and predicted prices
    #train = pd.DataFrame(data_filtered_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    #valid = pd.DataFrame(data_filtered_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    #valid.insert(1, "y_pred", y_pred, True)
    #valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
    #df_union = pd.concat([train, valid])
    ## Zoom in to a closer timeframe
    #df_union_zoom = df_union[df_union.index > display_start_date]
    ## Create the lineplot
    #fig, ax1 = plt.subplots(figsize=(16, 8))
    #plt.title("y_pred vs y_test")
    #plt.ylabel(COMPANY, fontsize=18)
    #sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    #sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)
    ## Create the bar plot with the differences
    #df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom["residuals"].dropna()]
    #ax1.bar(height=df_union_zoom['residuals'].dropna(), x=df_union_zoom['residuals'].dropna().index, width=3, label='residuals', color=df_sub)
    #plt.legend()
    #plt.show()
    #df_temp = trainData[-sequence_length:]
    #new_df = df_temp.filter(FEATURE_COLUNMS)
    #N = sequence_length
    ## Get the last N day closing price values and scale the data to be values between 0 and 1
    #last_N_days = new_df[-sequence_length:].values
    #last_N_days_scaled = scaler.transform(last_N_days)
    ## Create an empty list and Append past N days
    #X_test_new = []
    #X_test_new.append(last_N_days_scaled)
    ## Convert the X_test data set to a numpy array and reshape the data
    #pred_price_scaled = model.predict(np.array(X_test_new))
    #pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))
    ## Print last price and predicted price for the next day
    #price_today = np.round(new_df['Close'][-1], 2)
    #predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
    #change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)
    #plus = '+'; minus = ''
    #print(f'The close price for {COMPANY} at {TEST_END} was {price_today}')
    #print(f'The predicted close price is {predicted_price} ({plus if change_percent > 0 else minus}{change_percent}%)')
    #df_Close.to_csv("trainfilename.csv")

def createModel(layer_num, layer_size, layer_name, dropout):
    #Declare some variables so the model knows whats what
    PRICE_VALUE = "Close"

    


    
    




    #df_extract = trainData.filter([FEATURE_COLUNMS], axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(trainData[PRICE_VALUE].values.reshape(-1, 1)) 
    #DF = pd.DataFrame(scaled_data)
    #DF.to_csv("scaled_data.csv")

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
    multi_pred = multivariate_prediction(LAYER_NUM, LAYER_SIZE, LAYER_NAME)
    #createModel2(layer_num, layer_size, layer_name, dropout):
    model = createModel(LAYER_NUM, LAYER_SIZE, LAYER_NAME, DROPOUT)

    #Make sure it knows that testData is refering to the global
    global testData

    PRICE_VALUE = "Close"

    PREDICT_COLUNM = "Close"

    FEATURE_COLUNMS = "Open", "High", "Low", "Close", "Adj Close", "Volume"

    scaler = MinMaxScaler(feature_range=(0, 1)) 

    scaled_data = scaler.fit_transform(trainData[PRICE_VALUE].values.reshape(-1, 1)) 

    

    #------------------------------------------------------------------------------
    # Test the model accuracy on existing data
    #------------------------------------------------------------------------------
    # Load the test data
    #TEST_START = '2020-01-02'
    #TEST_END = '2022-12-31'

    #test_data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)
    #
    #test_data_filename = os.path.join("data", f"{COMPANY}_{TRAIN_START}_{TRAIN_END}.csv")
    #
    #testData = checkFiles(test_data_filename)

    # The above bug is the reason for the following line of code
    testData = testData[1:]

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
    if (SCALER):
        predicted_prices = scaler.inverse_transform(predicted_prices)
    # Clearly, as we transform our data into the normalized range (0,1),
    # we now need to reverse this transformation 

    #------------------------------------------------------------------------------
    # Predict next day
    #------------------------------------------------------------------------------

    futurePrice = []


    i = 0
 
    while i < PREDICTION_DAYS:
        i += 1

        real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
 
        prediction = model.predict(real_data)

        scaledPrdiction = scaler.inverse_transform(prediction)

        futurePrice.append(scaledPrdiction.flatten()[0])

        prediction = prediction.flatten()[0]

        print(futurePrice)

        model_inputs = pd.DataFrame(model_inputs)
        model_inputs.loc['0'] = prediction
        model_inputs = model_inputs.to_numpy()

        





        # make it unscales prediction data

        # make it so it redoes the declaromg of real_data
 
        # make it so it adds to real_data

        # male it so it rescales the real_data

        #futurePrice.append()

    #LINK https://stackoverflow.com/questions/69785891/how-to-use-the-lstm-model-for-multi-step-forecasting/69787683#69787683

    df_futurePrices = pd.DataFrame(columns=['Index','Forecast'])
    DF = pd.DataFrame(predicted_prices)
    df_futurePrices['Index'] = range(DF.index[-1] + 1, DF.index[-1] + 1 + PREDICTION_DAYS)
    df_futurePrices = df_futurePrices.set_index("Index")
    df_futurePrices['Forecast'] = np.array(futurePrice)

    #DF = pd.DataFrame(df_futurePrices)
    #DF.to_csv("data9.csv")

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

    
    #multi_pred.to_csv("data9.csv")
    #print(multi_pred.columns)
    #multi_pred.rename(columns={multi_pred.columns[1]: 'Close'},inplace=True)
    #print(multi_pred.columns)
    #multi_pred['Close'] = np.array(multi_pred)

    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.plot(df_futurePrices, color="orange", label=f"Predicted {COMPANY} Future Price")
    plt.plot(multi_pred, color="red", label=f"Predicted {COMPANY} multivariate Price")
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
    