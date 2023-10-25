
# evaluate prophet time series forecasting model on hold out dataset
from pandas import read_csv
from pandas import to_datetime
import pandas as pd
from pandas import DataFrame
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
# load data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
df = read_csv('daily-total-female-births.csv', header=0)
df.dropna(inplace=True)
df["Date"] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# prepare expected column names
df.columns = ['ds', 'y']
df['ds']= to_datetime(df['ds'])

#df.to_csv("data1.csv")
# create test dataset, remove last 12 months
train = df.drop(df.index[-1:])
train.to_csv("TRAINPRPFET.csv")
df.to_csv("TESTPRPFET.csv")
print(train.tail())
# define the model
model = Prophet()
# fit the model
model.fit(train)
# define the period for which we want a prediction
future = list()
future = df['ds']#.values
future = DataFrame(future)
future.to_csv("datafuture.csv")
# use the model to make a forecast
forecast = model.predict(future)
forecast.to_csv("dataforecast.csv")
# calculate MAE between expected and predicted values for december
y_true = df['y'][-len(forecast):].values
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)
# plot expected vs actual
pyplot.plot(y_true, label='Actual')
pyplot.plot(y_pred, label='Predicted')
pyplot.legend()
pyplot.show()