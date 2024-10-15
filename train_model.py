import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


file_path = 'Assignment-3-ML-Sales_Transactions_Dataset_Weekly.csv'
sales_data = pd.read_csv(file_path)


weeks_columns = [f'W{i}' for i in range(52)]
sales_long = sales_data.melt(id_vars=['Product_Code'], value_vars=weeks_columns, 
                             var_name='Week', value_name='Sales')
sales_long['Week_Num'] = sales_long['Week'].str.extract('(\d+)').astype(int)
sales_long = sales_long.sort_values(['Product_Code', 'Week_Num'])
product_sales = sales_long[sales_long['Product_Code'] == 'P1'].set_index('Week_Num')['Sales']

train_size = int(len(product_sales) * 0.8)
train_data, test_data = product_sales[:train_size], product_sales[train_size:]

model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()

predictions = model_fit.forecast(steps=len(test_data))

plt.figure(figsize=(10,6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Test Data', color='green')
plt.plot(test_data.index, predictions, label='Predictions', color='red')
plt.legend()
plt.title('Sales Forecasting using ARIMA')
plt.xlabel('Weeks')
plt.ylabel('Sales')
plt.show()

future_forecast = model_fit.forecast(steps=104)

print("Sales Forecast for the next 2 years (in weeks):")
print(future_forecast)










