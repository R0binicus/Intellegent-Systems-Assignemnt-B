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
SPLIT_DATE_BOOL = True

RATIO = 4      #Int or Float    #Not actually a ration, but idk what else to call it
RATIO_BOOL = False               #2 is train/test equally split, 4 is train gets about 75% of data

#If both SPLIT_DATE_BOOL and RATIO_BOOL are false, it picks a random date

STOREFILE = True                   # Pick whether to store file or not

SCALER = False                   # Pick whether to scale feature colunms or not

# Train and test data global variables for setting
trainData = None
testData = None