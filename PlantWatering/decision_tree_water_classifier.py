import ssl
import urllib.request
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score

# Download dataset from Google Sheet
plant_data_url = 'https://docs.google.com/spreadsheets/d/1X60NHE99UIrPwy4Cq6FfxNOW5XR-W2FzTrG2vAUYIhA/export?format=csv'
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

# Fix typo: soil_mositure  → soil_moisture
raw_df = plant_df[['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'plant_size',
                   'Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                   'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                   'Portulaca grandiflora', 'Pothos', 'Snake Plant', 'need_water']]

# Train‑val‑test split
train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

print('train_df.shape: ', train_df.shape)
print('val_df.shape: ', val_df.shape)
print('test_df.shape: ', test_df.shape)

input_cols = list(train_df.columns)[0:-1]
target_col = 'need_water'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

num_cols = ['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'plant_size']

# Min‑Max scaling
scaler = MinMaxScaler().fit(raw_df[num_cols])
train_inputs[num_cols] = scaler.transform(train_inputs[num_cols])
val_inputs[num_cols] = scaler.transform(val_inputs[num_cols])
test_inputs[num_cols] = scaler.transform(test_inputs[num_cols])

# Train decision tree with max_depth=4 (prevent overfit)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(train_inputs, train_targets)

print(f"Train Accuracy: {model.score(train_inputs, train_targets):.3f}")
print(f"Val Accuracy:   {model.score(val_inputs, val_targets):.3f}")

# Visualize decision tree
plt.figure(figsize=(20, 12))
plot_tree(model, feature_names=train_inputs.columns, filled=True, rounded=True, class_names=model.classes_)
plt.show()

# Function for error vs max‑depth analysis
def max_depth_error(maxDepth):
    model = DecisionTreeClassifier(max_depth=maxDepth, random_state=42)
    model.fit(train_inputs, train_targets)
    train_error = 1 - model.score(train_inputs, train_targets)
    val_error = 1 - model.score(val_inputs, val_targets)
    return {'Max Depth': maxDepth, 'Training Error': train_error, 'Validation Error': val_error}
