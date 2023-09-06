from keras.layers import LSTM, SimpleRNN, GRU


#
#
#
# --------------------Parameters-----------------------------------------
#
#
#

DATA_SOURCE = "yahoo"
COMPANY = "TSLA"

TRAIN_START = '2015-01-01'      #Start date of dataset    #Must be in 'YYYY-MM-DD' format eg '2015-01-01'
TEST_END = '2022-12-31'         #End date of dataset    #Must be in 'YYYY-MM-DD' format eg '2022-12-31'

SPLIT_DATE = '2020-01-01'       #Split date of dataset    #Must be in 'YYYY-MM-DD' format eg '2020-01-01'

RATIO = 4      #Int or Float    #Not actually a ration, but idk what else to call it
                                #2 is train/test equally split, 4 is train gets about 75% of data

MODE = 2
# 1 = Split dataset into train/test sets by date, then predict
# 2 = Split dataset into train/test sets by ratio, then predict
# 3 = Make candlestick chart of data from past NDAYS
# 4 = Make boxplot chart of data from past NDAYS
# Other = Split dataset into train/test sets randomly, then predict

NDAYS = 30                      # Set how many days to make a chart from

STOREFILE = True                # Pick whether to store file or not

SCALER = True                   # Pick whether to scale feature colunms or not


#
#
#
# --------------------MODEL SETTINGS-----------------------------------------
#
#
#

# Default Parameters

LAYER_NUM = 2

LAYER_SIZE = 50

LAYER_NAME = SimpleRNN

DROPOUT = 0.2


# Hyperparameters ------  Preset model parameters to use
HYPERPARAM = 5            # 0 means default, 1 means base LSTM, 2 means base RNN, 3 means base GRU,
                          # 4 means P1 settings, 5 means my custom settings

match HYPERPARAM:
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
        LAYER_NUM = 3
        LAYER_SIZE = 100
        LAYER_NAME = GRU
        DROPOUT = 0.3
    case _: #Default settings
        LAYER_NUM = LAYER_NUM # dummy, default  already declared values up above so doing nothing uses them



