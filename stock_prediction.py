# File: stock_prediction.py
# Authors: Robin Findlay-Marks
# Date: 26/10/2023

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import mplfinance as mpl 
import os
import random

from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, InputLayer, SimpleRNN, GRU
from keras.callbacks import EarlyStopping
from pandas import concat
from numpy import asarray
from prophet import Prophet
from os.path import exists
from datetime import timedelta
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from parameters import *
#from math import sqrt


# Train, test and full data global variables for setting later
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
        # NaN values from pandas are values that are not a number. For example in stocks if the stock data for a 
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

# This function gets the datafile name as well as the split date
# it then runs the file checker to get the dataset, then splits the dataset at the split date
# and sets the trainData and testData
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
    
    # Set the fulldata variable
    global fullData
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

# This function gets the datafile name as well as the 'ratio' number
# it then runs the file checker to get the dataset, then splits the dataset at the split date
# and sets the trainData and testData
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

    # Set the fulldata variable
    global fullData
    fullData = df

    # create train/test partition
    global trainData
    trainData = df[TRAIN_START:trainEndDate]
    trainData = trainData.drop(trainData.columns[[0]], axis=1)
    global testData
    testData = df[testStartDate:TEST_END]
    testData = testData.drop(testData.columns[[0]], axis=1)

    # Function for running ARIMA or SARIMA predictions
def ARIMA_prediction():
    # assign train and test data to variables
    train = trainData['Close'].values
    test1 = testData[1:]
    test = test1['Close'].values
    history = [x for x in train]
    predictions = list()
    # parameters for SARIMA
    my_seasonal_order = (SAUTOREG, SDIFERENCE, SMOVAVG, SEASON)

    # walk-forward validation
    for t in range(len(test)):
        # re-create the ARIMA model after each new observation 
        if SARIMA:
            model = ARIMA(history, order=(AUTOREG,DIFERENCE,MOVAVG), seasonal_order=my_seasonal_order)
        else:
            model = ARIMA(history, order=(AUTOREG,DIFERENCE,MOVAVG))
        model_fit = model.fit()
        # make prediction
        output = model_fit.forecast()
        forecast = output[0]
        predictions.append(forecast)
        expected = test[t]
        # keep track of past observations
        history.append(expected)
        print('predicted=%f, expected=%f' % (forecast, expected))
    return predictions

    # Function for running the multivariate prediction
def multivariate_prediction(layer_num, layer_size, layer_name):
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

    # Function for making the LSTM, GRU or SimpleRNN models
def createModelRNN(layer_num, layer_size, layer_name, dropout):
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


    # turn time series data into a supervised learning data
def data_to_supervised(data):
    df = pd.DataFrame(data)
    colunms = list()

    # sliding window technique used to make the new samples for the supervised learning data
    # input sequence (t-n, ... t-1)
    for i in range(1, 0, -1):
        colunms.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, 1):
        colunms.append(df.shift(-i))
    # put it all together
    newdf = concat(colunms, axis=1)
    # drop NaN values
    newdf.dropna(inplace=True)
    return newdf.values

    # fit an random forest model and make a one step prediction
def forest_prediction(train, testX):
    # turn list into an array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # make model
    model = RandomForestRegressor(n_estimators=FOREST_ESTIMATORS)
    model.fit(trainX, trainy)
    # make a single prediction
    forestPrediction = model.predict([testX])
    return forestPrediction[0]

    # loop for running the prediction on a number of days
def forest_loop(test, train):
    predictions = list()
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX = test[i, :-1]
        # fit model on history and make a prediction
        forestPrediction = forest_prediction(history, testX)
        # store forecast in list of predictions
        predictions.append(forestPrediction)
    return predictions

    # main function to run the rest of the random forest sub functions
def runTestForest():
    # get test + train data and make local variables
    global testData
    global trainData
    train = trainData["Close"].values
    test = testData["Close"].values

    # turn train and test data into supervised learning
    train = data_to_supervised(train)
    test = data_to_supervised(test)

    # make predictions
    forestPrediction = forest_loop(test, train)

    return forestPrediction

    # function to run the prohpet prediction
def runTestProphet():
    #Get pre-split data 
    # Only need to extract Close because the date is the index, and as such is automatically transfered over too
    test = fullData['Close']
    # reset index because it turns it back into a normal colunm which is referenced later
    test = test.reset_index()
    # turn date and close colunm into ds and y (necessary for prophet to work)
    test.columns = ['ds', 'y']
    # make sure ds colunm is a datatime
    test['ds']= pd.to_datetime(test['ds'])
    # remove offset days from the training data
    train = test.drop(test.index[-PROPHET_TRAIN_OFFSET:])
    
    # make the prophet model
    model = Prophet()
    # train the model from the train data
    model.fit(train)
    # setup dataframe from only the date date
    futuredays = list()
    futuredays = test['ds']
    futuredays = pd.DataFrame(futuredays)
    
    # use the model to make a forecast from the futuredays datetime range
    forecast = model.predict(futuredays)
    # set the actual and predicted values
    actualData = test['y'][-len(forecast):].values
    prophetPrediction = forecast['yhat'].values
    # plot actual vs Predicted Data
    plt.plot(actualData, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(prophetPrediction, color="green", label=f"Predicted {COMPANY} Price using Prophet")
    plt.legend()
    plt.show()

    # function for running most of the prediction functions. It starts the ARIMA_prediction (ensemble) and multivariate_prediction 
    # functions but the code is further above. This function runs most of the prediction for the LSTM, GRU or SimpleRNN models that
    # other than what is in the createModelRNN function. It then also does some post-processing on the data so it can all be 
    # displayed on a graph and finally plots and shows the graph
def runTest():
    if ENSEMBLE:
        arima_pred = ARIMA_prediction()
    if MULTIVARIATE:
        multi_pred = multivariate_prediction(LAYER_NUM, LAYER_SIZE, LAYER_NAME)
    
    model = createModelRNN(LAYER_NUM, LAYER_SIZE, LAYER_NAME, DROPOUT)

    #Make sure it knows that testData is refering to the global
    global testData
    PRICE_VALUE = "Close"
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(trainData[PRICE_VALUE].values.reshape(-1, 1)) 

    #------------------------------------------------------------------------------
    # Test the model accuracy on existing data
    #------------------------------------------------------------------------------

    testData = testData[1:]
    actual_prices = testData[PRICE_VALUE].values
    total_dataset = pd.concat((trainData[PRICE_VALUE], testData[PRICE_VALUE]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(testData) - LOOKBACK_DAYS:].values
    # We need to do the above because to predict the closing price of the fisrt
    # LOOKBACK_DAYS of the test period [TEST_START, TEST_END], we'll need the 
    # data from the training period

    model_inputs = model_inputs.reshape(-1, 1)

    model_inputs = scaler.transform(model_inputs)
    # We again normalize our closing price data to fit them into the range (0,1)
    # using the same scaler used above 

    #------------------------------------------------------------------------------
    # Make predictions on test data
    #------------------------------------------------------------------------------
    x_test = []
    for x in range(LOOKBACK_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - LOOKBACK_DAYS:x, 0])

    # set x_test to be an nparray, then reshape it
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

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


    # combine the RNN prediction value and the S/ARIMA prediction value and average them to make an ensemble value
    ensemble_preds = None
    if ENSEMBLE:
        predicted_prices_list = predicted_prices.tolist()
        i = 0
        # for each predicted value from each mode, add the two together and to make the ensemble value
        for value in arima_pred:
            # first parameter is the array that the second parameter is being added to
            ensemble_preds = np.append(ensemble_preds, (arima_pred[i] + predicted_prices_list[i])/2)
            i += 1

    # plot ensemble data if set to in parameters
    if ENSEMBLE:
        plt.plot(ensemble_preds, color="green", label=f"Ensemble Predicted {COMPANY} Price")
    else:
        plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
        plt.plot(df_futurePrices, color="orange", label=f"Predicted {COMPANY} Future Price")
        if MULTIVARIATE: # Display Multivariate data if it is enabled
            plt.plot(df_multiPrices, color="red", label=f"Predicted {COMPANY} multivariate Price")
    # plot forest data if set to in parameters
    if FOREST: # Display Forest data if it is enabled
        forestPrediction = runTestForest()
        plt.plot(forestPrediction, color="purple", label=f"Predicted {COMPANY} forest Price")
    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.show()

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

    # Add axis labels and display the plot
    plt.xlabel("Stock Data Type")
    plt.ylabel("Price")
    plt.show()

def Main(): #Main function for deciding what functions to use
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

        case 5: #Prophet Prediction
            getDataRatio(ticker_data_filename, RATIO)
            runTestProphet()

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
    