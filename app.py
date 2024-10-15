from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
      
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

   
        forecast_image = process_and_forecast(filepath)
        
   
        return render_template('result.html', forecast_image=forecast_image)


def process_and_forecast(filepath):
  
    sales_data = pd.read_csv(filepath)

   
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

    forecast_image = 'static/forecast.png'
    plt.savefig(forecast_image)
    plt.close()

    return forecast_image

if __name__ == '__main__':
  
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    

    app.run(debug=True)

    app.run(debug=True)




