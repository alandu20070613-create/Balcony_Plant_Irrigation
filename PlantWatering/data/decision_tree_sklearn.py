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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree, export_text

plant_data_url = 'https://docs.google.com/spreadsheets/d/1X60NHE99UIrPwy4Cq6FfxNOW5XR-W2FzTrG2vAUYIhA/export?format=csv'

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

# One Hot Encoder encodes plant type and add into the table (better method for more elements in array)
enc = preprocessing.OneHotEncoder()
enc.fit(plant_df[['plant_type']]) # fit plant_type into an array
one_hot = enc.transform(plant_df[['plant_type']]).toarray()
plant_df[['Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
       'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
       'Portulaca grandiflora', 'Pothos', 'Snake Plant']] = one_hot
# print(plant_df)

# split the data into train, verify and test sets
raw_df = plant_df[['pot_volume', 'temperature', 'humidity', 'soil_mositure ', 'light_intensity', 'plant_size',
                   'Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                   'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                   'Portulaca grandiflora', 'Pothos', 'Snake Plant', 'need_water']]
train_val_df, test_df = train_test_split(raw_df, test_size = 0.2, random_state = 42) # uses 42 as seed in random number generator
train_df, val_df = train_test_split(train_val_df, test_size = 0.25, random_state = 42)

print('train_df.shape: ', train_df.shape)
print('val_df.shape: ', val_df.shape)
print('test_df.shape: ', test_df.shape)

# create input and target columns, split them based on train-val-test sets and make copies
input_cols = list(train_df.columns)[0:-1] # exclude the need_water column
target_col = 'need_water'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# list numerical and categorical columns
num_cols = ['pot_volume', 'temperature', 'humidity', 'soil_mositure ', 'light_intensity', 'plant_size']
categorical_cols = ['Aloe Vera', 'Bougainvillea', 'Hydrangea', 'Jasmine', 'Lavender',
                    'Mint', 'Monstera', 'Peace Lily', 'Petunia', 'Phalaenopsis',
                    'Portulaca grandiflora', 'Pothos', 'Snake Plant']


# scale down numerical values into 0-1 and check by printing min/max
print(val_inputs.describe().loc[['min', 'max']])

scaler = MinMaxScaler().fit(raw_df[num_cols])
train_inputs[num_cols] = scaler.transform(train_inputs[num_cols])
val_inputs[num_cols] = scaler.transform(val_inputs[num_cols])
test_inputs[num_cols] = scaler.transform(test_inputs[num_cols])

print(val_inputs.describe().loc[['min', 'max']])

"""
# train the model using a decision tree
model = DecisionTreeClassifier(random_state=42)
model.fit(train_inputs, train_targets)

# make predictions
train_predictions = model.predict(train_inputs) # using train_inputs data to predict
# print(train_predictions)
# print(pd.Series(train_predictions).value_counts()) # turn numpy array into Series and then count # of yes/no values
# print(train_targets.value_counts()) # count # of yes/no values of train_targets

# calculate an accuracy score and probability seeing the similarity of the two data and confidence of predictions
# print(accuracy_score(train_predictions, train_targets))
train_probabilities = model.predict_proba(train_inputs)
# print(train_probabilities)

# calculate the accuracy score of val sets based on the trained model from above
print(model.score(val_inputs, val_targets))
print(val_targets.value_counts() / len(val_targets)) # how many times yes/no appeared divided by total numbers, the probability of choosing Yes all the time
"""

"""
# show Decision Tree Diagram
plt.figure(figsize = (20, 12))
plot_tree(model, feature_names = train_inputs.columns, max_depth = 2, filled=True)
plt.show()
"""

# print(model.tree_.max_depth) # print the depth of the tree

"""
# show decision tree in text form
tree_text = export_text(model, max_depth = 7, feature_names = list(train_inputs.columns))
print(tree_text)
"""

"""
# show importance of each variable
print(model.feature_importances_)
importance_df = pd.DataFrame({'feature': train_inputs.columns,
                              'importance': model.feature_importances_}).sort_values('importance', ascending=False)
plt.title('Decision Tree Feature Importance', color='white')
plt.tick_params(colors='white')
sns.barplot(x = 'importance', y = 'feature', data = importance_df, color='blue')
plt.show()
"""

# choose a max depth of 4 to avoid overfitting (boosting val accuracy) and show the graph
model = DecisionTreeClassifier(max_depth = 4, random_state=42)
model.fit(train_inputs, train_targets)
print(model.score(train_inputs, train_targets))
print(model.score(val_inputs, val_targets))

plt.figure(figsize = (20, 12))
plot_tree(model, feature_names = train_inputs.columns, filled=True, rounded=True, class_names=model.classes_)
plt.show()

def max_depth_error(maxDepth):
    model = DecisionTreeClassifier(max_depth = maxDepth, random_state=42)
    model.fit(train_inputs, train_targets)
    train_error = 1 - model.score(train_inputs, train_targets)
    val_error = 1 - model.score(val_inputs, val_targets)
    return {'Max Depth' : maxDepth, 'Training Error' : train_error, 'Validation Error' : val_error}



