from keras.layers import LSTM, RNN, GRU


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

LAYER_NUM = 2

LAYER_SIZE = 50

LAYER_NAME = LSTM

DROPOUT = 0.2