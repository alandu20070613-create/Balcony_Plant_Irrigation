import ssl
from urllib.request import urlretrieve
import pandas as pd
import urllib.request
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from narwhals.selectors import categorical
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

plant_data_url = 'https://docs.google.com/spreadsheets/d/1ElclVQ1kTJnPLNpz3OyQ9wf8zfz8CpmjcnUVRGRqFKo/export?format=csv'

# Create unverified SSL
ssl_context = ssl._create_unverified_context()

with urllib.request.urlopen(plant_data_url, context = ssl_context) as response:
    with open('plant_data.csv', 'wb') as out_file:
        out_file.write(response.read())

# read local file
plant_df = pd.read_csv('plant_data.csv')
# plant_df.info()             # check columns, non-null count, and data types
# print(plant_df.describe())  # check statistical information

# setup to create histogram and boxplot
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# histogram and boxplot for temperature

plant_df.temperature.describe()
fig = px.histogram(plant_df, x='temperature', marginal='box', nbins=50, title='Distribution of Temperature')
fig.update_layout(bargap=0.1)
fig.show()

# histogram and boxplot for humidity
plant_df.temperature.describe()
fig = px.histogram(plant_df, x='humidity', marginal='box', color_discrete_sequence=['orange'], title='Distribution of Humidity')
fig.update_layout(bargap=0.1)
fig.show()

# histogram and boxplot for water amount based on plant size
plant_df.temperature.describe()
fig = px.histogram(plant_df, x='water_amount', marginal='box', color='plant_size', color_discrete_sequence=['red', 'green', 'blue'], title='Water_Amount')
fig.update_layout(bargap=0.1)
fig.show()


# print(plant_df.plant_size.value_counts())  # check the number of small, medium and large plant


# scatterplot for light intensity and water amount based on plant type
fig = px.scatter(plant_df, x='light_intensity', y='water_amount', color='plant_type', opacity=0.8, title='Light Intensity vs. Water Amount')
fig.update_traces(marker_size=5)
fig.show()


# correlation between water amount and variables
print(plant_df.water_amount.corr(plant_df.temperature))
print(plant_df.water_amount.corr(plant_df.humidity))
print(plant_df.water_amount.corr(plant_df.pot_volume))
print(plant_df.water_amount.corr(plant_df.soil_moisture))
print(plant_df.water_amount.corr(plant_df.light_intensity))
"""

"""
# turn categorical data (plant type) into numerical data
plant_type_values = {'Aloe Vera' : 1,
                     'Bougainvillea' : 2,
                     'Hydrangea' : 3,
                     'Jasmine' : 4,
                     'Lavender' : 5,
                     'Mint' : 6,
                     'Monstera' : 7,
                     'Peace Lily' : 8,
                     'Petunia' : 9,
                     'Phalaenopsis' : 10,
                     'Portulaca grandiflora' : 11,
                     'Pothos' : 12,
                     'Snake Plant' : 13}
plant_type_numeric = plant_df.plant_type.map(plant_type_values)
print(plant_type_numeric)
print(plant_df.water_amount.corr(plant_type_numeric))
"""

"""
# show all correlations in columns without categorical data (plant type)
columns_to_exclude = ['plant_type']
numeric_df = plant_df.drop(columns=columns_to_exclude)
print(numeric_df.corr())

# show all correlations in heatmap form
sns.heatmap(numeric_df.corr(), cmap = 'Reds', annot = True)
plt.title('Correlation Heatmap')
plt.show()
"""


# show pot volume and water amount relationship specifically for small plant size
small_plant_size_df = plant_df[plant_df.plant_size == 0]
"""
plt.title('Pot Volume vs. Water Amount')
sns.scatterplot(data = small_plant_size_df, x = 'pot_volume', y = 'water_amount', alpha = 0.7, s = 15)
plt.show()
"""


# define a function for estimated water amount
def estimate_water_amount(pot_volume, w, b):
    return w * pot_volume + b

"""
# compare estimated water amount and actual water amount for small plants' pot volume
print(small_plant_size_df.pot_volume)
print(estimate_water_amount(small_plant_size_df.pot_volume, 10, 30))
print(small_plant_size_df.water_amount)
"""

"""
# plot a linear line between small plants' pot volume and estimated water amount
plt.plot(small_plant_size_df.pot_volume, estimate_water_amount(small_plant_size_df.pot_volume, 10, 30), 'pink')
plt.xlabel('Pot Volume')
plt.ylabel('Water Amount')
plt.show()
"""

"""
# plot scatterplot(actual data) and linear line (estimated line)
plt.plot(small_plant_size_df.pot_volume, estimate_water_amount(small_plant_size_df.pot_volume, 10, 30), 'pink', alpha = 0.9, label = 'Estimate')
plt.scatter(small_plant_size_df.pot_volume, small_plant_size_df.water_amount, alpha = 0.7, label='Actual')
plt.xlabel('Pot Volume')
plt.ylabel('Water Amount')
plt.legend()
plt.show()
"""


# define a function that allows the users to try different w, b parameters and print graph
def try_parameters(w, b):
    plt.plot(small_plant_size_df.pot_volume, estimate_water_amount(small_plant_size_df.pot_volume, w, b), 'pink', alpha=0.9, label='Estimate')
    plt.scatter(small_plant_size_df.pot_volume, small_plant_size_df.water_amount, alpha=0.7, label='Actual')
    plt.xlabel('Pot Volume')
    plt.ylabel('Water Amount')
    plt.legend()
    plt.show()

"""
try_parameters(8, 35)
"""

"""
# define a loss function that calculates how good is the prediction
targets = small_plant_size_df.water_amount
predictions = estimate_water_amount(small_plant_size_df.pot_volume, 10, 30)
"""
def rmse(target, prediction):
    return np.sqrt(np.mean(np.square(target - prediction)))

"""
print(rmse(targets, predictions))
"""

"""
# define a function that allows the users to try different w, b parameters and calculate rmse
targets = small_plant_size_df.water_amount
predictions = estimate_water_amount(small_plant_size_df.pot_volume, 10, 30)

def try_parameters(w, b):
    plt.plot(small_plant_size_df.pot_volume, estimate_water_amount(small_plant_size_df.pot_volume, w, b), 'pink', alpha=0.9, label='Estimate')
    plt.scatter(small_plant_size_df.pot_volume, small_plant_size_df.water_amount, alpha=0.7, label='Actual')
    plt.xlabel('Pot Volume')
    plt.ylabel('Water Amount')
    plt.legend()
    plt.show()
    loss = rmse(targets, predictions)
    print("RMSE Loss: ", loss)

try_parameters(12, 30)
"""

"""
# using linear regression to predict water amount using pot volume data
model = LinearRegression()
help(model.fit)
inputs = plant_df[['pot_volume']] # input (x values) must be 2D
targets = plant_df.water_amount # y values can be 1D
print('inputs.shape:', inputs.shape)
print('targets.shape:', targets.shape)

model.fit(inputs, targets)
new_data = pd.DataFrame({
    'pot_volume': [7.5, 3.5, 9.8]
})
print(model.predict(new_data)) # predict water amount when pot volume = 7.5, 3.5, 9.8
print(model.predict(inputs)) # predict all 206 values
print(targets)

print(rmse(targets, model.predict(inputs))) # find rmse loss function
print(model.coef_) # find predicted w value
print(model.intercept_) # fine predicted b value
try_parameters(model.coef_, model.intercept_) # show the best-fit (linear regression) line and scatterplot 
"""

"""
# enumerate plant types and add into the table
plant_type_values = {'Aloe Vera' : 1,
                     'Bougainvillea' : 2,
                     'Hydrangea' : 3,
                     'Jasmine' : 4,
                     'Lavender' : 5,
                     'Mint' : 6,
                     'Monstera' : 7,
                     'Peace Lily' : 8,
                     'Petunia' : 9,
                     'Phalaenopsis' : 10,
                     'Portulaca grandiflora' : 11,
                     'Pothos' : 12,
                     'Snake Plant' : 13}
plant_df['plant_type_values'] = plant_df.plant_type.map(plant_type_values)
print(plant_df)
"""

# One Hot Encoder encodes plant type and add into the table (better method for more elements in array)
enc = preprocessing.OneHotEncoder()
enc.fit(plant_df[['plant_type']]) # fit plant_type into an array
one_hot = enc.transform(plant_df[['plant_type']]).toarray()
plant_df[['Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
       'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
       'Portulaca grandiflora', 'Pothos', 'Snake Plant']] = one_hot
print(plant_df)


# using multi-variable linear regression to predict water amount using pot volume data
inputs = plant_df[['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'plant_size',
                   'Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                   'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                   'Portulaca grandiflora', 'Pothos', 'Snake Plant']]
targets = plant_df.water_amount
model = LinearRegression().fit(inputs, targets)
print(model.predict(inputs))
print('Loss:', rmse(targets, model.predict(inputs)))

# print(model.predict([[12.2, 36.5, 78, 45, 555, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])) # make a prediction

"""
# Barplot for plant type and water amount
sns.barplot(data = plant_df, x = 'plant_type', y = 'water_amount')
plt.title('Plant Type vs. Water Amount')
plt.show()
"""

"""
# weights of each variable without standardizing
weights_df = pd.DataFrame({
    'feature' : np.append(inputs.columns, 1),
    'weight' : np.append(model.coef_, model.intercept_)
})
print(weights_df)


# find mean and variance for the numeric variables
num_cols = ['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity']
scaler = StandardScaler()
scaler.fit(plant_df[num_cols])

print(scaler.mean_)
print(scaler.var_ )

# standardize the variables into normal distribution values
numerical_cols = scaler.transform(plant_df[num_cols])
print(numerical_cols)

# turn dataframe into a numpy array
cat_cols = ['Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
       'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
       'Portulaca grandiflora', 'Pothos', 'Snake Plant']
categorical_cols = plant_df[cat_cols].values

all_inputs = np.hstack([numerical_cols, categorical_cols]) # combine numerical and categorical data

# weights of each variable with standardizing
weights_df = pd.DataFrame({
    'feature' : np.append(inputs.columns, 1),
    'weight' : np.append(model.coef_, model.intercept_)
})
print(weights_df.sort_values('weight', ascending=False))

# Predict the new plant water amount after standardizing (same value)
new_plant =[[12.2, 36.5, 78, 45, 555, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
print(scaler.transform([[12.2, 36.5, 78, 45, 555]]))
print(model.predict(new_plant))
"""

"""
# use a portion of the data
use_sample = False
sample_fraction = 0.2
if use_sample:
    raw_df = plant_df.sample(frac = sample_fraction)



# split the data into train, verify and test sets
raw_df = plant_df[['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'plant_size',
                   'Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                   'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                   'Portulaca grandiflora', 'Pothos', 'Snake Plant', 'water_amount']]
train_val_df, test_df = train_test_split(raw_df, test_size = 0.2, random_state = 42) # uses 42 as seed in random number generator
train_df, val_df = train_test_split(train_val_df, test_size = 0.25, random_state = 42)

print('train_df.shape: ', train_df.shape)
print('val_df.shape: ', val_df.shape)
print('test_df.shape: ', test_df.shape)

# create input and target columns, split them based on train-val-test sets and make copies
input_cols = list(train_df.columns)[1:-1] # exclude the water amount column
target_col = 'water_amount'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# list numerical and categorical columns
num_cols = ['pot_volume', 'temperature', 'humidity', 'soil_moisture', 'light_intensity', 'plant_size']
categorical_cols = ['Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                    'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                    'Portulaca grandiflora', 'Pothos', 'Snake Plant']








