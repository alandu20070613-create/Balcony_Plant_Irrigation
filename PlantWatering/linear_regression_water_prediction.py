import ssl
import urllib.request
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Download dataset
plant_data_url = 'https://docs.google.com/spreadsheets/d/1ElclVQ1kTJnPLNpz3OyQ9wf8zfz8CpmjcnUVRGRqFKo/export?format=csv'
ssl_context = ssl._create_unverified_context()

with urllib.request.urlopen(plant_data_url, context=ssl_context) as response:
    with open('plant_data.csv', 'wb') as out_file:
        out_file.write(response.read())

plant_df = pd.read_csv('plant_data.csv')

# Plot styling
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# One‑Hot encode plant_type
enc = preprocessing.OneHotEncoder()
enc.fit(plant_df[['plant_type']])
one_hot = enc.transform(plant_df[['plant_type']]).toarray()
plant_df[['Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
        'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
        'Portulaca grandiflora', 'Pothos', 'Snake Plant']] = one_hot

# Helper functions
def estimate_water_amount(pot_volume, w, b):
    return w * pot_volume + b

def rmse(target, prediction):
    return np.sqrt(np.mean(np.square(target - prediction)))

small_plant_size_df = plant_df[plant_df.plant_size == 0]

def try_parameters(w, b):
    plt.plot(small_plant_size_df.pot_volume, estimate_water_amount(small_plant_size_df.pot_volume, w, b),
             'pink', alpha=0.9, label='Estimate')
    plt.scatter(small_plant_size_df.pot_volume, small_plant_size_df.water_amount, alpha=0.7, label='Actual')
    plt.xlabel('Pot Volume')
    plt.ylabel('Water Amount')
    plt.legend()
    plt.show()

# Multi‑variable linear regression
inputs = plant_df[['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'plant_size',
                   'Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                   'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                   'Portulaca grandiflora', 'Pothos', 'Snake Plant']]
targets = plant_df.water_amount

model = LinearRegression().fit(inputs, targets)
print("Predictions:", model.predict(inputs))
print(f"RMSE Loss: {rmse(targets, model.predict(inputs)):.3f}")

# Standard scaling for feature weight analysis
num_cols = ['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity']
scaler = StandardScaler()
scaler.fit(plant_df[num_cols])
numerical_cols = scaler.transform(plant_df[num_cols])
cat_cols = ['Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
            'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
            'Portulaca grandiflora', 'Pothos', 'Snake Plant']
categorical_cols = plant_df[cat_cols].values
all_inputs = np.hstack([numerical_cols, categorical_cols])

weights_df = pd.DataFrame({
    'feature': np.append(inputs.columns, "intercept"),
    'weight': np.append(model.coef_, model.intercept_)
})
print(weights_df.sort_values('weight', ascending=False))

# Train‑val‑test split for regression
raw_df = plant_df[['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'plant_size',
                   'Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                   'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                   'Portulaca grandiflora', 'Pothos', 'Snake Plant', 'water_amount']]

train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

print('train_df.shape: ', train_df.shape)
print('val_df.shape: ', val_df.shape)
print('test_df.shape: ', test_df.shape)

input_cols = list(train_df.columns)[1:-1]
target_col = 'water_amount'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()
