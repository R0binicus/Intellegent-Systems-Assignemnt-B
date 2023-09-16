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
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, InputLayer, SimpleRNN, GRU

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

    # create train/test partition
    global trainData
    trainData = df[TRAIN_START:splitDate]
    global testData
    testData = df[testStartDate:TEST_END]
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

    # create train/test partition
    global trainData
    trainData = df[TRAIN_START:trainEndDate]
    global testData
    testData = df[testStartDate:TEST_END]

    #trainData.to_csv("trainfilename.csv")
    #testData.to_csv("testfilename.csv")        #test that the data is split correctly

    # construct the model
    # model = create_model(N_STEPS, len(FEATURE_COLUMNS), units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, 
    #                 loss=LOSS,  optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    #(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
    #            loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):

    #(LOOKBACK_DAYS, n_features, units=50, cell=LSTM, n_layers=2, dropout=0.2,
    #            loss="mean_squared_error", optimizer="adam", bidirectional=False):

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

    model.compile(optimizer='adam', loss='mean_squared_error')
    # Now we are going to train this model with our training data 
    # (x_train, y_train)
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Return completed model to be tested 
    return model

def runTest():
    #createModel2(layer_num, layer_size, layer_name, dropout):
    model = createModel(LAYER_NUM, LAYER_SIZE, LAYER_NAME, DROPOUT)

    #Make sure it knows that testData is refering to the global
    global testData

    PRICE_VALUE = "Close"

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

    DF = pd.DataFrame(df_futurePrices)
    DF.to_csv("data9.csv")

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
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.plot(df_futurePrices, color="orange", label=f"Predicted {COMPANY} Future Price")
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


def createModel2():
    # For more details: 
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
    #------------------------------------------------------------------------------
    # Prepare Data
    ## To do:
    # 1) Check if data has been prepared before. 
    # If so, load the saved data
    # If not, save the data into a directory
    # 2) Use a different price value eg. mid-point of Open & Close
    # 3) Change the Prediction days
    #------------------------------------------------------------------------------
    PRICE_VALUE = "Close"

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    # Note that, by default, feature_range=(0, 1). Thus, if you want a different 
    # feature_range (min,max) then you'll need to specify it here
    scaled_data = scaler.fit_transform(trainData[PRICE_VALUE].values.reshape(-1, 1)) 
    # Flatten and normalise the data
    # First, we reshape a 1D array(n) to 2D array(n,1)
    # We have to do that because sklearn.preprocessing.fit_transform()
    # requires a 2D array
    # Here n == len(scaled_data)
    # Then, we scale the whole array to the range (0,1)
    # The parameter -1 allows (np.)reshape to figure out the array size n automatically 
    # values.reshape(-1, 1) 
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
    # When reshaping an array, the new shape must contain the same number of elements 
    # as the old shape, meaning the products of the two shapes' dimensions must be equal. 
    # When using a -1, the dimension corresponding to the -1 will be the product of 
    # the dimensions of the original array divided by the product of the dimensions 
    # given to reshape so as to maintain the same number of elements.

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

    #------------------------------------------------------------------------------
    # Build the Model
    ## TO DO:
    # 1) Check if data has been built before. 
    # If so, load the saved data
    # If not, save the data into a directory
    # 2) Change the model to increase accuracy?
    #------------------------------------------------------------------------------
    model = Sequential() # Basic neural network
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    # for some useful examples

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # This is our first hidden layer which also spcifies an input layer. 
    # That's why we specify the input shape for this layer; 
    # i.e. the format of each training example
    # The above would be equivalent to the following two lines of code:
    # model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
    # model.add(LSTM(units=50, return_sequences=True))
    # For some advances explanation of return_sequences:
    # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
    # https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
    # As explained there, for a stacked LSTM, you must set return_sequences=True 
    # when stacking LSTM layers so that the next LSTM layer has a 
    # three-dimensional sequence input. 

    # Finally, units specifies the number of nodes in this layer.
    # This is one of the parameters you want to play with to see what number
    # of units will give you better prediction quality (for your problem)

    model.add(Dropout(0.2))
    # The Dropout layer randomly sets input units to 0 with a frequency of 
    # rate (= 0.2 above) at each step during training time, which helps 
    # prevent overfitting (one of the major problems of ML). 

    model.add(LSTM(units=50, return_sequences=True))
    # More on Stacked LSTM:
    # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1)) 
    # Prediction of the next closing value of the stock price

    # We compile the model by specify the parameters for the model
    # See lecture Week 6 (COS30018)
    model.compile(optimizer='adam', loss='mean_squared_error')
    # The optimizer and loss are two important parameters when building an 
    # ANN model. Choosing a different optimizer/loss can affect the prediction
    # quality significantly. You should try other settings to learn; e.g.

    # optimizer='rmsprop'/'sgd'/'adadelta'/...
    # loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

    # Now we are going to train this model with our training data 
    # (x_train, y_train)
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    return model
    # Other parameters to consider: How many rounds(epochs) are we going to 
    # train our model? Typically, the more the better, but be careful about
    # overfitting!
    # What about batch_size? Well, again, please refer to 
    # Lecture Week 6 (COS30018): If you update your model for each and every 
    # input sample, then there are potentially 2 issues: 1. If you training 
    # data is very big (billions of input samples) then it will take VERY long;
    # 2. Each and every input can immediately makes changes to your model
    # (a souce of overfitting). Thus, we do this in batches: We'll look at
    # the aggreated errors/losses from a batch of, say, 32 input samples
    # and update our model based on this aggregated loss.

    # TO DO:
    # Save the model and reload it
    # Sometimes, it takes a lot of effort to train your model (again, look at
    # a training data with billions of input samples). Thus, after spending so 
    # much computing power to train your model, you may want to save it so that
    # in the future, when you want to make the prediction, you only need to load
    # your pre-trained model and run it on the new input for which the prediction
    # need to be made.
    