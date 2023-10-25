#import warnings
#warnings.filterwarnings("ignore")
#from ts_utils import *

from pandas import read_csv
from pandas import to_datetime
import pandas as pd
from pandas import DataFrame
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot

dtf = pd.read_csv('data_sales.csv')
dtf.head()
dtf["date"] = pd.to_datetime(dtf['date'], format='%d.%m.%Y')
ts = dtf.groupby("date")["item_cnt_day"].sum().rename("sales")
ts.head()
ts.tail()
print("population --> len:", len(ts), "| mean:", round(ts.mean()), " | std:", round(ts.std()))
w = 30
print("moving --> len:", w, " | mean:", round(ts.ewm(span=w).mean()[-1]), " | std:", round(ts.ewm(span=w).std()[-1]))
plot_ts(ts, plot_ma=True, plot_intervals=True, window=w, figsize=(15,5))