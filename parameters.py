from keras.layers import LSTM, SimpleRNN, GRU


#
#
#
# --------------------Parameters-----------------------------------------
#
#
#

DATA_SOURCE = "yahoo"
COMPANY = "TSLA"                # Which comany's ticker symbol to use for the program

TRAIN_START = '2016-06-01'      #Start date of dataset    #Must be in 'YYYY-MM-DD' format eg '2015-01-01'
TEST_END = '2023-09-30'         #End date of dataset    #Must be in 'YYYY-MM-DD' format eg '2022-12-31'

SPLIT_DATE = '2022-01-01'       #Split date of dataset    #Must be in 'YYYY-MM-DD' format eg '2022-01-01'
    # WARNING must be on 2022-01-01 or later otherwise i don't think there is enough data to predict properly

PREDICTION_DAYS = 1             # Number of days into the future to predict the stock prices

LOOKBACK_DAYS = 60              #Number of days to look back to base the prediction
                                # 60 Original

RATIO = 5      #Int or Float    #Not actually a ration, but idk what else to call it
                                #2 is train/test equally split, 4 is train gets about 75% of data

FEATURE_COLUNMS = 'Open','High','Low','Close','Adj Close','Volume'

MODE = 5
# 1 = Split dataset into train/test sets by date, then predict
# 2 = Split dataset into train/test sets by ratio, then predict
# 3 = Make candlestick chart of data from past NDAYS
# 4 = Make boxplot chart of data from past NDAYS
# 5 = Run Prediction with Prophet model
# Other = Split dataset into train/test sets randomly, then predict

STOREFILE = True                # Pick whether to store file or not

NDAYS = 30                      # Set how many days in the future to make a predictions in the future

SCALER = True                   # Pick whether to predict and display Multivariate data in additional to other

MULTIVARIATE = False            # Pick whether to predict and display Multivariate data in additional to other

ENSEMBLE = False                 # Pick whether to use ensemble with arima/sarima or not
SARIMA = False                   # True for SARIMA false for ARIMA

FOREST = False                   # True to make and display Random Forest Predictions
FOREST_ESTIMATORS = 10          # The number of estimators the Random Forest Model uses 

PROPHET_TRAIN_OFFSET = 300      # The number of days to take off the end of the train data

#
#
#
# --------------------ADVANCED SETTINGS-----------------------------------------
#
#
#

# Default Parameters

#RNN param 
LAYER_NUM = 2       # number of layers used   
LAYER_SIZE = 50     # size of each layer
LAYER_NAME = LSTM   # which type of RNN model is used
DROPOUT = 0.2       # dropout

#ARIMA param
AUTOREG = 5     #Trend autoregression order
DIFERENCE = 1   #Trend difference order
MOVAVG = 0      #Trend moving average order

#SARIMA param
SAUTOREG = 1    #Seasonal autoregressive order
SDIFERENCE = 1  #Seasonal difference order
SMOVAVG = 0     #Seasonal moving average order
SEASON = 6      #The number of time steps for a single seasonal period

# Hyperparameters

RNN_HYPERPARAM = 5
ARIMA_HYPERPARAM = 1
SARIMA_HYPERPARAM = 1

match RNN_HYPERPARAM:
    case 1: #LSTM
        LAYER_NUM = 2
        LAYER_SIZE = 50
        LAYER_NAME = LSTM
        DROPOUT = 0.2
    case 2: # RNN
        LAYER_NUM = 2
        LAYER_SIZE = 50
        LAYER_NAME = SimpleRNN
        DROPOUT = 0.2
    case 3: #GRU 
        LAYER_NUM = 2
        LAYER_SIZE = 50
        LAYER_NAME = GRU
        DROPOUT = 0.2
    case 4: #P1 settings
        LAYER_NUM = 2
        LAYER_SIZE = 256
        LAYER_NAME = LSTM
        DROPOUT = 0.4
    case 5: #Custom
        LAYER_NUM = 2
        LAYER_SIZE = 50
        LAYER_NAME = GRU
        DROPOUT = 0.1
    case _: #Default settings
        LAYER_NUM = 2
        LAYER_SIZE = 50
        LAYER_NAME = SimpleRNN
        DROPOUT = 0.2

match ARIMA_HYPERPARAM:
    case 1: 
        AUTOREG = 5
        DIFERENCE = 1
        MOVAVG = 0
    case 2: 
        AUTOREG = 10
        DIFERENCE = 1
        MOVAVG = 0
    case 3: 
        AUTOREG = 5
        DIFERENCE = 10
        MOVAVG = 0
    case 4: 
        AUTOREG = 5
        DIFERENCE = 1
        MOVAVG = 1
    case 5: 
        AUTOREG = 1
        DIFERENCE = 1
        MOVAVG = 1
    case _: #Default settings
        AUTOREG = 5
        DIFERENCE = 1
        MOVAVG = 0

match SARIMA_HYPERPARAM:
    case 1: #7 day week
        SAUTOREG = 1    
        SDIFERENCE = 1  
        SMOVAVG = 0     
        SEASON = 7         
    case 2: #30 day month
        SAUTOREG = 1    
        SDIFERENCE = 1  
        SMOVAVG = 0     
        SEASON = 30         
    case 3: # 365 day year
        SAUTOREG = 1    
        SDIFERENCE = 1  
        SMOVAVG = 0     
        SEASON = 365            
    case _: #Default settings
        SAUTOREG = 1    
        SDIFERENCE = 1  
        SMOVAVG = 0     
        SEASON = 7      