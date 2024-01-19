#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

# XML file upload
xml_file_path = '/Users/melihakbulut/Desktop/dışa aktarılan.xml'  
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Convert datas into pandas DataFrame 
records = []
for record in root.findall('.//Record'):
    attributes = record.attrib
    # Just take 'HKQuantityTypeIdentifierActiveEnergyBurned' type records
    if attributes.get('type') == 'HKQuantityTypeIdentifierActiveEnergyBurned':
        date_str = attributes.get('creationDate').split(' ')[0]
        date = datetime.strptime(date_str, '%Y-%m-%d')
        calories = float(attributes.get('value'))
        records.append({'Date': date, 'Calories': calories})

# DataFrame create
df = pd.DataFrame(records)

# CSV convert
csv_file_path = 'exported_calories_data.csv' 
df.to_csv(csv_file_path, index=False)

# View data
print(df)


# In[ ]:


import pandas as pd

# CSV file upload
csv_file_path = 'exported_calories_data.csv'  
df = pd.read_csv(csv_file_path)

# Convert date column to datetime type 
df['Date'] = pd.to_datetime(df['Date'])

# Getting year information by date
df['Year'] = df['Date'].dt.year

# Calculating seasons by date
def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Fall'
    else:
        return 'Winter'

# Calculate season using date in each row
df['Season'] = df['Date'].apply(get_season)

# Group data by year and season
yearly_seasonal_data = df.groupby(['Year', 'Season']).size().unstack().fillna(0)

# Show the resulting output
print(yearly_seasonal_data)


# In[ ]:


import pandas as pd

# CSV file upload
csv_file_path = 'exported_calories_data.csv'  
df = pd.read_csv(csv_file_path)

# Convert date column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Calculating seasons by date
def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Fall'
    else:
        return 'Winter'

# Calculate season using date in each row
df['Season'] = df['Date'].apply(get_season)

# Group and average data by season
seasonal_avg_data = df.groupby('Season')['Calories'].mean()


print(seasonal_avg_data)


# In[ ]:


import pandas as pd

# CSV file upload
csv_file_path = 'exported_calories_data.csv'  
df = pd.read_csv(csv_file_path)

# Convert date column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Calculating seasons by date
def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Fall'
    else:
        return 'Winter'

# calculate seasons using date in each row
df['Season'] = df['Date'].apply(get_season)

# Group and average data by season
seasonal_avg_data = df.groupby('Season')['Calories'].mean().reset_index()

# Save data as CSV
output_csv_path = 'seasonal_average_calories.csv'  
seasonal_avg_data.to_csv(output_csv_path, index=False)

print(f"Seasonal average calories data saved to {output_csv_path}")


# In[ ]:


import matplotlib.pyplot as plt

# Create separate histograms for each season
for season in df['Season'].unique():
    seasonal_data = df[df['Season'] == season]
    plt.figure()
    plt.hist(seasonal_data['Calories'], bins=20, alpha=0.7)
    plt.xlabel('Calories')
    plt.ylabel('Frequency')
    plt.title(f'Calorie Distribution for the {season} Season')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Create a line graph showing the average calorie trend based on seasonal variables
# Calculate the average calories for each season
seasonal_avg_data = df.groupby('Season')['Calories'].mean().reset_index()

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(seasonal_avg_data['Season'], seasonal_avg_data['Calories'], marker='o', linestyle='-')
plt.xlabel('Season')
plt.ylabel('Average Calories')
plt.title('Seasonal Average Calorie Trend')
plt.grid(True)
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a box plot to compare calorie distributions based on seasons
plt.figure(figsize=(10, 6))
sns.boxplot(x='Season', y='Calories', data=df)
plt.xlabel('Season')
plt.ylabel('Calories')
plt.title('Seasonal Comparative Box Plot of Calorie Distributions')
plt.grid(True)
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

# Generate sample data for calories
start_date = date(2022, 1, 1)
end_date = date(2024, 1, 1)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]

data = {
    'Date': date_range,
    'Calories': [random.uniform(0, 5000) for _ in range(len(date_range))]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a 'Season' column based on the month of each date
df['Month'] = df['Date'].apply(lambda x: x.month)
seasons = {
    1: 'Winter',
    2: 'Winter',
    3: 'Spring',
    4: 'Spring',
    5: 'Spring',
    6: 'Summer',
    7: 'Summer',
    8: 'Summer',
    9: 'Fall',
    10: 'Fall',
    11: 'Fall',
    12: 'Winter'
}
df['Season'] = df['Month'].map(seasons)

# Save the DataFrame as a CSV file
df.to_csv('seasonal_calories.csv', index=False)

  


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and preprocess data
data = pd.read_csv('seasonal_calories.csv')  
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Seasonal decomposition
result = seasonal_decompose(data['Calories'], model='additive', period=365)  

# Visualize seasonal components
result.plot()
plt.show()

# ACF and PACF plots
plot_acf(data['Calories'])
plot_pacf(data['Calories'])
plt.show()

# Train and forecast SARIMA model
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 365)
model = SARIMAX(data['Calories'], order=order, seasonal_order=seasonal_order)
model_fit = model.fit()

# Make predictions
forecast_steps = 365  # Forecast for the next 365 days
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# Visualize forecasts and confidence interval
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Calories'], label='Real Data')
plt.plot(forecast_mean.index, forecast_mean.values, color='red', label='Forecast')
plt.fill_between(forecast_mean.index, forecast.conf_int()['lower Calories'], forecast.conf_int()['upper Calories'], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Calories')
plt.title('Seasonal Calorie Trend')
plt.legend()
plt.show()


# In[ ]:




