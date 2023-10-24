
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import sklearn
from statsmodels.tsa.arima.model import ARIMA

buildings = pd.read_csv("building_metadata.csv")
df = pd.read_csv('train.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[df['meter'] == 3].reset_index(drop=True) # 3 corresponds to the hot water meter

buildings['primary_use'].value_counts()


building_id = 1266
sector = buildings[buildings['building_id'] == building_id]['primary_use'].values[0]
building_1034_data = df[df['building_id'] == building_id][['timestamp','meter','meter_reading']].sort_values('timestamp').reset_index(drop=True)

train, valid = building_1034_data[:-int(len(building_1034_data)*0.10)], building_1034_data[-int(len(building_1034_data)*0.10):]

nrows, ncols = 1, 1
fig, ax = plt.subplots(nrows, ncols, figsize=(16,4))

#ax1 = plt.subplot(nrows, ncols, 1)
#ax1.plot(train['timestamp'].values, train['meter_reading'].values, c='tab:olive')
#ax1.plot(valid['timestamp'].values, valid['meter_reading'].values, c='tab:purple')
#ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
#ax1.set_ylabel('Hot Water Usage')
#ax1.set_xlabel('Timestamp')
#ax1.set_title('Building ID: {}, Sector: {}'.format(building_id, sector), fontsize=10)
#
#plt.tight_layout()
#plt.show()


def add_timestamp_features(data):
    pd.options.mode.chained_assignment = None

    data['day'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    data['is_month_start'] = data['timestamp'].dt.is_month_start
    data['is_month_end'] = data['timestamp'].dt.is_month_end

    conditions = [(data['dayofweek'].eq(5) | data['dayofweek'].eq(6))]
    choices = [1]
    data['is_weekend'] = np.select(conditions, choices, default=0)
    return data

train = add_timestamp_features(train)

#nrows, ncols = 6,1 
#fig, ax = plt.subplots(nrows, ncols, figsize=(16,10))
#
#for i, col in enumerate(['day','hour','dayofweek','is_month_start','is_month_end','is_weekend']):
#    
#    plot_data = train.groupby(col)['meter_reading'].mean()
#    
#    cax = plt.subplot(nrows, ncols, i+1)
#    cax.bar(plot_data.index, plot_data.values)
#    # Need to add automatic adding of labels of x-axis
#    cax.set_title('{}'.format(col, fontsize=10))
#
#plt.tight_layout()
#plt.show()

pd.options.mode.chained_assignment = 'warn'




valid = add_timestamp_features(valid)

x_train, y_train = train.drop(columns=['meter_reading']), train['meter_reading'].values
x_valid, y_valid = valid.drop(columns=['meter_reading']), valid['meter_reading'].values

params = {'num_leaves': 30,
          'n_estimators': 400,
          'max_depth': 8,
          'min_child_samples': 200,
          'learning_rate': 0.1,
          'subsample': 0.50,
          'colsample_bytree': 0.75
         }

model = lgb.LGBMRegressor(**params)
model = model.fit(x_train.drop(columns=['timestamp']), y_train)

def plot_predictions(x_valid, y_valid, valid_preds, building_id, sector):
    
    rmse = sklearn.metrics.mean_squared_error(y_valid, valid_preds, squared=False)
    
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(16,4))

    ax1 = plt.subplot(nrows, ncols, 1)
    ax1.plot(x_valid['timestamp'].values, y_valid, c='tab:purple')
    ax1.scatter(x_valid['timestamp'].values, valid_preds, s=5, c='#7FB285')
    ax1.set_ylabel('Hot Water Usage')
    ax1.set_xlabel('Timestamp')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax1.set_title('Prediction RMSE: {:.4f}'.format(rmse), fontsize=10)

    plt.tight_layout()
    plt.show()
    
valid_preds = model.predict(x_valid.drop(columns=['timestamp']))    
#plot_predictions(x_valid, y_valid, valid_preds, building_id, sector)















steps_ahead = 10
sarima_model = ARIMA(y_train[-250:-steps_ahead], order=(1,1,1), seasonal_order=(0, 1, 2, 24)).fit()
sarima_valid_preds = []

for i in range(len(y_train) - steps_ahead, len(y_train)):
    sarima_model = sarima_model.append([y_train[i]])
    sarima_valid_preds.append(sarima_model.forecast(steps_ahead)[0])
    
for i in range(len(y_valid) - steps_ahead):
    sarima_model = sarima_model.append([y_valid[i]])
    sarima_valid_preds.append(sarima_model.forecast(steps_ahead)[0])
    

rmse = sklearn.metrics.mean_squared_error(y_valid, sarima_valid_preds, squared=False)

nrows, ncols = 1, 1
fig, ax = plt.subplots(nrows, ncols, figsize=(16,4))

ax1 = plt.subplot(nrows, ncols, 1)
ax1.plot(x_valid['timestamp'].values, y_valid, c='tab:purple')
ax1.scatter(x_valid['timestamp'].values, sarima_valid_preds, s=5, c='#7FB285')
ax1.set_ylabel('Hot Water Usage')
ax1.set_xlabel('Timestamp')
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
ax1.set_title('Prediction RMSE: {:.4f}'.format(rmse), fontsize=10)

plt.tight_layout()
plt.show()





ensemble_preds = np.add(valid_preds, sarima_valid_preds)/2
rmse = sklearn.metrics.mean_squared_error(y_valid, ensemble_preds, squared=False)

nrows, ncols = 1, 1
fig, ax = plt.subplots(nrows, ncols, figsize=(16,4))

ax1 = plt.subplot(nrows, ncols, 1)
ax1.plot(x_valid['timestamp'].values, y_valid, c='tab:purple')
ax1.scatter(x_valid['timestamp'].values, ensemble_preds, s=5, c='#7FB285')
ax1.set_ylabel('Hot Water Usage')
ax1.set_xlabel('Timestamp')
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
ax1.set_title('Prediction RMSE: {:.4f}'.format(rmse), fontsize=10)

plt.tight_layout()
plt.show()


dummy = "dummy"

#https://towardsdatascience.com/time-series-forecasting-ensemble-learning-df5fcbb48581