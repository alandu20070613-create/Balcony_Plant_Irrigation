# Plant‑Irrigation‑ML‑Analysis

> 
> EDA & Machine Learning for Smart Plant Irrigation Prediction

Machine learning project to analyze plant environmental data, predict required watering volume and classify whether plants need water.

Code Inspiration from freeCodeCamp.org


## Features

- **Exploratory Data Analysis**: Histograms, boxplots, scatter plots, correlation heatmap
- **Regression**: Multivariate linear regression to predict `water_amount`
- **Classification**: Decision Tree classifier for `need_water` prediction, tune `max_depth` to prevent overfitting
- Data auto‑download from Google Sheets, One‑Hot encoding, feature scaling, RMSE loss calculation

## Requirements

```
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
narwhals
```

## 🚀 Quick Start

```
git clone https://github.com/your-username/Plant‑Irrigation‑ML‑Analysis.git
cd Plant‑Irrigation‑ML‑Analysis
pip install -r requirements.txt
python plant_irrigation_analysis.py
```

- Automatically downloads dataset as `plant_data.csv`
- Output interactive Plotly & Matplotlib charts
- Prints metrics, RMSE and model scores to console

## Project Structure

```
├── plant_irrigation_analysis.py   # Main script
├── plant_data.csv                 # Auto‑generated dataset
├── requirements.txt
└── README.md
```

## Known Issues

1. Typo bug: `soil_mositure ` (extra space) → fix to `soil_moisture`
2. Unverified SSL context for development use only
3. Many commented‑out code blocks for learning / debugging

## Possible Improvements

- Save trained models with joblib
- Add cross‑validation
- Auto‑export plot images
- Clean missing data handling

> 
> For educational purposes only, no license specified.# Balcony_Plant_Irrigation
