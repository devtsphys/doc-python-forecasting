# Time Series Forecasting with Scikit-Learn Reference Card

## Table of Contents

1. [Core Imports](#core-imports)
1. [Data Preprocessing](#data-preprocessing)
1. [Feature Engineering](#feature-engineering)
1. [Model Selection](#model-selection)
1. [Hyperparameter Tuning](#hyperparameter-tuning)
1. [Evaluation Metrics](#evaluation-metrics)
1. [Complete Examples](#complete-examples)
1. [Methods Reference Table](#methods-reference-table)

## Core Imports

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
```

## Data Preprocessing

### Time Series Data Structure

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Sort by date
df = df.sort_index()

# Handle missing values
df = df.fillna(method='ffill')  # Forward fill
df = df.fillna(method='bfill')  # Backward fill
df = df.interpolate()           # Linear interpolation

# Resample data
df_daily = df.resample('D').mean()
df_weekly = df.resample('W').mean()
df_monthly = df.resample('M').mean()
```

### Stationarity Testing

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] < 0.05  # p-value < 0.05 means stationary
```

### Differencing

```python
# First difference
df['value_diff1'] = df['value'].diff()

# Second difference
df['value_diff2'] = df['value_diff1'].diff()

# Seasonal difference
df['value_seasonal_diff'] = df['value'].diff(12)  # 12 for monthly data
```

## Feature Engineering

### Lag Features

```python
def create_lag_features(df, target_col, lags):
    """Create lag features for time series"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Usage
lags = [1, 2, 3, 7, 14, 30]
df = create_lag_features(df, 'value', lags)
```

### Rolling Window Features

```python
def create_rolling_features(df, target_col, windows):
    """Create rolling window statistics"""
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
        df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window).median()
    return df

# Usage
windows = [3, 7, 14, 30]
df = create_rolling_features(df, 'value', windows)
```

### Exponential Moving Average

```python
def create_ema_features(df, target_col, alphas):
    """Create exponential moving average features"""
    for alpha in alphas:
        df[f'{target_col}_ema_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
    return df

# Usage
alphas = [0.1, 0.3, 0.5, 0.7]
df = create_ema_features(df, 'value', alphas)
```

### Time-based Features

```python
def create_time_features(df):
    """Create time-based features from datetime index"""
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    return df
```

### Fourier Features

```python
def create_fourier_features(df, period, order):
    """Create Fourier series features for seasonality"""
    t = np.arange(len(df))
    for i in range(1, order + 1):
        df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)
    return df

# Usage for yearly seasonality (365 days)
df = create_fourier_features(df, 365, 3)
```

### Target Encoding for Categorical Variables

```python
def target_encoding(df, cat_col, target_col, smoothing=1):
    """Target encoding with smoothing"""
    global_mean = df[target_col].mean()
    category_stats = df.groupby(cat_col)[target_col].agg(['mean', 'count']).reset_index()
    category_stats['encoded'] = (category_stats['mean'] * category_stats['count'] + 
                                global_mean * smoothing) / (category_stats['count'] + smoothing)
    
    encoding_map = dict(zip(category_stats[cat_col], category_stats['encoded']))
    df[f'{cat_col}_encoded'] = df[cat_col].map(encoding_map)
    return df
```

## Model Selection

### Time Series Cross-Validation

```python
# Time Series Split
tscv = TimeSeriesSplit(n_splits=5, test_size=30)

# Custom time series split
def time_series_split(df, n_splits=5, test_size=0.2):
    """Custom time series split"""
    total_size = len(df)
    test_size = int(total_size * test_size)
    train_size = total_size - test_size
    
    splits = []
    for i in range(n_splits):
        train_end = train_size - (n_splits - i - 1) * (test_size // n_splits)
        test_start = train_end
        test_end = test_start + test_size // n_splits
        
        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, min(test_end, total_size)))
        
        splits.append((train_indices, test_indices))
    
    return splits
```

### Model Dictionary

```python
models = {
    'linear_regression': LinearRegression(),
    'ridge': Ridge(),
    'lasso': Lasso(),
    'elastic_net': ElasticNet(),
    'random_forest': RandomForestRegressor(random_state=42),
    'gradient_boosting': GradientBoostingRegressor(random_state=42),
    'extra_trees': ExtraTreesRegressor(random_state=42),
    'svr': SVR(),
    'knn': KNeighborsRegressor(),
    'decision_tree': DecisionTreeRegressor(random_state=42)
}
```

## Hyperparameter Tuning

### Parameter Grids

```python
param_grids = {
    'ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'lasso': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'elastic_net': {
        'alpha': [0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'svr': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance']
    }
}
```

### Grid Search with Time Series CV

```python
def tune_model(model, param_grid, X, y, cv=None):
    """Tune model with time series cross-validation"""
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
```

## Evaluation Metrics

### Custom Scoring Functions

```python
def evaluate_model(y_true, y_pred):
    """Comprehensive model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'SMAPE': smape
    }

# Directional accuracy
def directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy"""
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction)
```

## Complete Examples

### Basic Time Series Forecasting Pipeline

```python
def time_series_pipeline(df, target_col, test_size=0.2):
    """Complete time series forecasting pipeline"""
    
    # 1. Feature Engineering
    df = create_time_features(df)
    df = create_lag_features(df, target_col, [1, 2, 3, 7])
    df = create_rolling_features(df, target_col, [3, 7, 14])
    df = create_ema_features(df, target_col, [0.1, 0.3])
    
    # 2. Remove NaN values
    df = df.dropna()
    
    # 3. Prepare features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # 4. Time series split
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = evaluate_model(y_test, y_pred)
    
    return results, scaler, models

# Usage
results, scaler, trained_models = time_series_pipeline(df, 'value')
```

### Advanced Pipeline with Feature Selection

```python
def advanced_pipeline(df, target_col, test_size=0.2, k_best=20):
    """Advanced pipeline with feature selection and tuning"""
    
    # Feature engineering
    df = create_time_features(df)
    df = create_lag_features(df, target_col, list(range(1, 15)))
    df = create_rolling_features(df, target_col, [3, 7, 14, 30])
    df = create_ema_features(df, target_col, [0.1, 0.2, 0.3, 0.5])
    df = create_fourier_features(df, 365, 3)
    
    # Remove NaN values
    df = df.dropna()
    
    # Prepare data
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # Time series split
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Create pipeline with feature selection and scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_regression, k=k_best)),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'selector__k': [10, 20, 30],
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, None]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    results = evaluate_model(y_test, y_pred)
    
    return best_model, results, grid_search.best_params_
```

### Walk-Forward Validation

```python
def walk_forward_validation(df, target_col, model, window_size=100, step_size=1):
    """Implement walk-forward validation"""
    
    predictions = []
    actuals = []
    
    for i in range(window_size, len(df), step_size):
        # Training data
        train_data = df.iloc[i-window_size:i]
        test_data = df.iloc[i:i+1]
        
        # Feature engineering for training data
        train_features = create_features(train_data, target_col)
        test_features = create_features(test_data, target_col)
        
        # Remove NaN and align features
        train_features = train_features.dropna()
        feature_cols = [col for col in train_features.columns if col != target_col]
        
        X_train = train_features[feature_cols]
        y_train = train_features[target_col]
        X_test = test_features[feature_cols]
        y_test = test_features[target_col]
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        predictions.extend(y_pred)
        actuals.extend(y_test.values)
    
    return np.array(predictions), np.array(actuals)

def create_features(df, target_col):
    """Helper function to create features"""
    df = df.copy()
    df = create_time_features(df)
    df = create_lag_features(df, target_col, [1, 2, 3])
    df = create_rolling_features(df, target_col, [3, 7])
    return df
```

## Methods Reference Table

|Category               |Method/Function              |Purpose                  |Key Parameters                 |
|-----------------------|-----------------------------|-------------------------|-------------------------------|
|**Data Preprocessing** |`pd.to_datetime()`           |Convert to datetime      |`format`, `errors`             |
|                       |`df.resample()`              |Resample time series     |`rule`, `how`                  |
|                       |`df.interpolate()`           |Fill missing values      |`method`, `limit`              |
|                       |`adfuller()`                 |Stationarity test        |`maxlag`, `regression`         |
|**Feature Engineering**|`df.shift()`                 |Create lag features      |`periods`, `fill_value`        |
|                       |`df.rolling()`               |Rolling window stats     |`window`, `min_periods`        |
|                       |`df.ewm()`                   |Exponential moving avg   |`alpha`, `span`                |
|                       |`df.diff()`                  |Differencing             |`periods`                      |
|**Models**             |`LinearRegression()`         |Linear regression        |`fit_intercept`, `normalize`   |
|                       |`Ridge()`                    |Ridge regression         |`alpha`, `solver`              |
|                       |`Lasso()`                    |Lasso regression         |`alpha`, `max_iter`            |
|                       |`RandomForestRegressor()`    |Random Forest            |`n_estimators`, `max_depth`    |
|                       |`GradientBoostingRegressor()`|Gradient Boosting        |`learning_rate`, `n_estimators`|
|                       |`SVR()`                      |Support Vector Regression|`C`, `gamma`, `epsilon`        |
|**Cross-Validation**   |`TimeSeriesSplit()`          |Time series CV           |`n_splits`, `test_size`        |
|                       |`GridSearchCV()`             |Hyperparameter tuning    |`param_grid`, `cv`, `scoring`  |
|**Scaling**            |`StandardScaler()`           |Standardization          |`with_mean`, `with_std`        |
|                       |`MinMaxScaler()`             |Min-max scaling          |`feature_range`                |
|**Feature Selection**  |`SelectKBest()`              |Select K best features   |`score_func`, `k`              |
|**Evaluation**         |`mean_squared_error()`       |MSE calculation          |`squared`                      |
|                       |`mean_absolute_error()`      |MAE calculation          |`multioutput`                  |
|                       |`r2_score()`                 |R² calculation           |`multioutput`                  |
|**Pipeline**           |`Pipeline()`                 |Create ML pipeline       |`steps`                        |

## Key Tips and Best Practices

### Feature Engineering Tips

- Always create lag features (1, 2, 3, 7, 14, 30 days)
- Use rolling statistics with different window sizes
- Include time-based features (month, day, weekday)
- Apply cyclical encoding for periodic features
- Consider seasonal differencing for seasonal data
- Use target encoding for categorical variables

### Model Selection Tips

- Start with simple models (Linear Regression, Ridge)
- Try ensemble methods (Random Forest, Gradient Boosting)
- Use time series cross-validation, not random splits
- Consider walk-forward validation for realistic evaluation
- Test multiple models and ensemble predictions

### Hyperparameter Tuning Tips

- Use TimeSeriesSplit for cross-validation
- Start with wide parameter ranges, then narrow down
- Use early stopping for gradient boosting models
- Consider Bayesian optimization for expensive models
- Always validate on out-of-sample data

### Common Pitfalls to Avoid

- Don’t use future information in features (data leakage)
- Don’t use random train/test splits
- Don’t ignore stationarity requirements
- Don’t forget to scale features for distance-based models
- Don’t overfit to validation data during tuning

This reference card provides a comprehensive toolkit for time series forecasting with scikit-learn, covering everything from basic preprocessing to advanced modeling techniques.


# Python Prophet Time Series Forecasting - Complete Reference Card

## Installation & Import

```python
# Installation
pip install prophet

# Import
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Basic Prophet Workflow

### 1. Data Preparation

```python
# Required format: DataFrame with 'ds' (datestamp) and 'y' (target) columns
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365, freq='D'),
    'y': np.random.randn(365).cumsum() + 100
})

# Ensure proper datetime format
df['ds'] = pd.to_datetime(df['ds'])
```

### 2. Model Creation & Training

```python
# Basic model
model = Prophet()
model.fit(df)

# Model with parameters
model = Prophet(
    growth='linear',           # 'linear' or 'logistic'
    seasonality_mode='additive',  # 'additive' or 'multiplicative'
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    mcmc_samples=0,
    interval_width=0.80,
    uncertainty_samples=1000
)
```

### 3. Forecasting

```python
# Create future dataframe
future = model.make_future_dataframe(periods=365, freq='D')

# Generate forecast
forecast = model.predict(future)

# Key forecast columns
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

## Core Prophet Parameters

|Parameter                |Type     |Default   |Description                               |
|-------------------------|---------|----------|------------------------------------------|
|`growth`                 |str      |‘linear’  |Growth model: ‘linear’, ‘logistic’, ‘flat’|
|`changepoints`           |list     |None      |List of dates for potential changepoints  |
|`n_changepoints`         |int      |25        |Number of potential changepoints          |
|`changepoint_range`      |float    |0.8       |Proportion of history for changepoints    |
|`yearly_seasonality`     |bool/int |‘auto’    |Yearly seasonal component                 |
|`weekly_seasonality`     |bool/int |‘auto’    |Weekly seasonal component                 |
|`daily_seasonality`      |bool/int |‘auto’    |Daily seasonal component                  |
|`holidays`               |DataFrame|None      |Holiday effects dataframe                 |
|`seasonality_mode`       |str      |‘additive’|‘additive’ or ‘multiplicative’            |
|`seasonality_prior_scale`|float    |10.0      |Controls seasonality flexibility          |
|`holidays_prior_scale`   |float    |10.0      |Controls holiday effects flexibility      |
|`changepoint_prior_scale`|float    |0.05      |Controls changepoint flexibility          |
|`mcmc_samples`           |int      |0         |MCMC sampling for uncertainty intervals   |
|`interval_width`         |float    |0.80      |Width of uncertainty intervals            |
|`uncertainty_samples`    |int      |1000      |Samples for uncertainty estimation        |

## Feature Engineering

### 1. Custom Seasonalities

```python
# Add custom seasonality
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5,
    prior_scale=10.0,
    mode='additive'  # or 'multiplicative'
)

# Conditional seasonality
def is_weekend(ds):
    date = pd.to_datetime(ds)
    return date.weekday() >= 5

model.add_seasonality(
    name='weekend_seasonality',
    period=1,
    fourier_order=3,
    condition_name='is_weekend'
)

# Add condition column to dataframes
df['is_weekend'] = df['ds'].apply(is_weekend)
future['is_weekend'] = future['ds'].apply(is_weekend)
```

### 2. Holidays and Events

```python
# Create holidays dataframe
holidays = pd.DataFrame({
    'holiday': 'thanksgiving',
    'ds': pd.to_datetime(['2020-11-26', '2021-11-25', '2022-11-24']),
    'lower_window': -1,  # day before
    'upper_window': 1,   # day after
})

# Built-in country holidays
from prophet.make_holidays import make_holidays_df
holidays_us = make_holidays_df(
    year_list=range(2020, 2025),
    country='US'
)

# Apply holidays
model = Prophet(holidays=holidays)
```

### 3. External Regressors

```python
# Add regressors (must be in both training and future dataframes)
model.add_regressor('temperature', prior_scale=10.0, mode='additive')
model.add_regressor('promotion', prior_scale=5.0, mode='multiplicative')

# Training data with regressors
df_with_regressors = df.copy()
df_with_regressors['temperature'] = np.random.normal(25, 5, len(df))
df_with_regressors['promotion'] = np.random.binomial(1, 0.1, len(df))

# Future data with regressors
future_with_regressors = model.make_future_dataframe(periods=30)
future_with_regressors['temperature'] = np.random.normal(25, 5, len(future_with_regressors))
future_with_regressors['promotion'] = np.random.binomial(1, 0.1, len(future_with_regressors))
```

### 4. Growth Models

#### Logistic Growth

```python
# Requires capacity (maximum value)
df['cap'] = 1000  # carrying capacity
future['cap'] = 1000

# Optional floor for logistic growth
df['floor'] = 10
future['floor'] = 10

model = Prophet(growth='logistic')
model.fit(df)
```

#### Piecewise Linear Growth

```python
# Custom changepoints
changepoints = ['2020-06-01', '2020-12-01']
model = Prophet(changepoints=changepoints)
```

## Method Reference Table

|Method                   |Description           |Parameters                                  |Returns                   |
|-------------------------|----------------------|--------------------------------------------|--------------------------|
|`Prophet()`              |Initialize model      |See parameters table above                  |Prophet object            |
|`fit(df)`                |Train the model       |`df`: DataFrame with ‘ds’, ‘y’              |Fitted Prophet object     |
|`predict(future)`        |Generate forecasts    |`future`: DataFrame with ‘ds’               |DataFrame with predictions|
|`make_future_dataframe()`|Create future dates   |`periods`, `freq`, `include_history`        |DataFrame                 |
|`add_seasonality()`      |Add custom seasonality|`name`, `period`, `fourier_order`, etc.     |None                      |
|`add_regressor()`        |Add external regressor|`name`, `prior_scale`, `standardize`, `mode`|None                      |
|`plot()`                 |Plot forecast         |`forecast`, `ax`, `uncertainty`, `plot_cap` |matplotlib axes           |
|`plot_components()`      |Plot components       |`forecast`, `uncertainty`, `plot_cap`       |matplotlib figure         |
|`cross_validation()`     |Time series CV        |`initial`, `period`, `horizon`              |DataFrame                 |
|`performance_metrics()`  |Calculate metrics     |`df_cv`, `metrics`                          |DataFrame                 |

## Model Diagnostics & Validation

### 1. Cross Validation

```python
from prophet.diagnostics import cross_validation, performance_metrics

# Time series cross-validation
df_cv = cross_validation(
    model, 
    initial='730 days',    # Initial training period
    period='180 days',     # Gap between cutoffs
    horizon='365 days'     # Forecast horizon
)

# Performance metrics
df_performance = performance_metrics(df_cv)
print(df_performance)
```

### 2. Hyperparameter Tuning

```python
import itertools

# Parameter grid
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

# Generate all combinations
all_params = [dict(zip(param_grid.keys(), v)) 
              for v in itertools.product(*param_grid.values())]

# Evaluation function
def evaluate_params(params, df):
    model = Prophet(**params)
    model.fit(df)
    df_cv = cross_validation(model, initial='365 days', 
                           period='90 days', horizon='90 days')
    df_p = performance_metrics(df_cv)
    return df_p['mape'].mean()

# Find best parameters
best_params = min(all_params, key=lambda x: evaluate_params(x, df))
```

### 3. Residual Analysis

```python
# Calculate residuals
forecast_train = model.predict(df)
residuals = df['y'] - forecast_train['yhat']

# Plot residuals
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['ds'], residuals)
plt.title('Residuals Over Time')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30)
plt.title('Residual Distribution')
plt.tight_layout()
```

## Advanced Techniques

### 1. Handling Outliers

```python
# Remove outliers before training
Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[(df['y'] >= lower_bound) & (df['y'] <= upper_bound)]
```

### 2. Multiple Time Series

```python
def forecast_multiple_series(df_dict):
    forecasts = {}
    for series_name, series_df in df_dict.items():
        model = Prophet()
        model.fit(series_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecasts[series_name] = forecast
    return forecasts
```

### 3. Feature Importance

```python
# Get regressor coefficients
def get_feature_importance(model, forecast):
    # Extract regressor coefficients from the model
    regressor_coefficients = {}
    for regressor in model.extra_regressors:
        # Get the coefficient from the fitted model
        coef = model.params[f'beta_{regressor}'][0] if f'beta_{regressor}' in model.params else 0
        regressor_coefficients[regressor] = coef
    return regressor_coefficients
```

### 4. Ensemble Methods

```python
def ensemble_forecast(df, n_models=5):
    forecasts = []
    
    for i in range(n_models):
        # Add noise to create model diversity
        df_noisy = df.copy()
        df_noisy['y'] += np.random.normal(0, df['y'].std() * 0.01, len(df))
        
        model = Prophet(
            changepoint_prior_scale=np.random.uniform(0.01, 0.5),
            seasonality_prior_scale=np.random.uniform(0.1, 10)
        )
        model.fit(df_noisy)
        
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecasts.append(forecast['yhat'].values)
    
    # Ensemble average
    ensemble_pred = np.mean(forecasts, axis=0)
    return ensemble_pred
```

## Evaluation Metrics

### Built-in Metrics

```python
# Available metrics in performance_metrics()
metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']
df_p = performance_metrics(df_cv, metrics=metrics)
```

### Custom Metrics

```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
```

## Common Patterns & Tips

### Data Frequency Patterns

```python
# Different frequencies
frequencies = {
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly': 'MS',  # Month start
    'Quarterly': 'QS',  # Quarter start
    'Yearly': 'YS',   # Year start
    'Hourly': 'H',
    'Business days': 'B'
}
```

### Troubleshooting Common Issues

1. **Insufficient Data**: Prophet needs at least 2 periods of seasonality
1. **Missing Values**: Prophet can handle missing values in ‘y’ but not ‘ds’
1. **Irregular Timestamps**: Use `freq` parameter in `make_future_dataframe()`
1. **Scale Issues**: Prophet works best with data scaled between 0-100
1. **Overfitting**: Reduce `changepoint_prior_scale` if model is too flexible

### Performance Optimization

```python
# For faster fitting (less accurate uncertainty intervals)
model = Prophet(
    mcmc_samples=0,          # Disable MCMC
    uncertainty_samples=100   # Reduce uncertainty samples
)

# For production deployment
model = Prophet(
    daily_seasonality=False,  # Disable if not needed
    weekly_seasonality='auto',
    yearly_seasonality='auto'
)
```

## Complete Example Pipeline

```python
# Full pipeline example
def prophet_pipeline(df, forecast_days=30):
    # 1. Data validation
    assert 'ds' in df.columns and 'y' in df.columns
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 2. Model setup with tuned parameters
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # 3. Add custom components
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # 4. Fit model
    model.fit(df)
    
    # 5. Generate forecast
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # 6. Model evaluation
    df_cv = cross_validation(model, initial='365 days', 
                           period='90 days', horizon='30 days')
    df_performance = performance_metrics(df_cv)
    
    return model, forecast, df_performance

# Usage
model, forecast, performance = prophet_pipeline(df, 30)
print(f"MAPE: {performance['mape'].mean():.2f}%")
```

This reference card covers all essential Prophet functionality for time series forecasting, from basic usage to advanced techniques and optimization strategies.


# Python Keras Forecasting Reference Card

## Core Imports & Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Bidirectional, Attention, MultiHeadAttention, TimeDistributed
```

## Data Preprocessing Functions

### Time Series Data Preparation

```python
def create_sequences(data, seq_length, target_col=0):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col])
    return np.array(X), np.array(y)

def split_time_series(data, train_ratio=0.8):
    """Split time series data maintaining temporal order"""
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def scale_data(train_data, test_data, scaler_type='minmax'):
    """Scale time series data"""
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler
```

## Model Architecture Components

|Layer Type            |Purpose               |Key Parameters                        |Example                           |
|----------------------|----------------------|--------------------------------------|----------------------------------|
|**LSTM**              |Long-term dependencies|`units`, `return_sequences`, `dropout`|`LSTM(50, return_sequences=True)` |
|**GRU**               |Simplified LSTM       |`units`, `return_sequences`, `dropout`|`GRU(32, return_sequences=False)` |
|**SimpleRNN**         |Basic recurrent       |`units`, `activation`                 |`SimpleRNN(64, activation='tanh')`|
|**Conv1D**            |Feature extraction    |`filters`, `kernel_size`, `activation`|`Conv1D(64, 3, activation='relu')`|
|**Dense**             |Fully connected       |`units`, `activation`                 |`Dense(1, activation='linear')`   |
|**Dropout**           |Regularization        |`rate`                                |`Dropout(0.2)`                    |
|**BatchNormalization**|Normalization         |`momentum`, `epsilon`                 |`BatchNormalization()`            |

## Common Model Architectures

### 1. Simple LSTM Model

```python
def create_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### 2. Bidirectional LSTM

```python
def create_bidirectional_lstm(input_shape, units=50):
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(units)),
        Dropout(0.2),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### 3. CNN-LSTM Hybrid

```python
def create_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### 4. GRU with Attention

```python
def create_gru_attention_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    gru = layers.GRU(64, return_sequences=True)(inputs)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(gru, gru)
    concat = layers.Concatenate()([gru, attention])
    pooling = layers.GlobalAveragePooling1D()(concat)
    dense = layers.Dense(50, activation='relu')(pooling)
    outputs = layers.Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### 5. Encoder-Decoder (Seq2Seq)

```python
def create_encoder_decoder(input_shape, forecast_horizon):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    encoder = layers.LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = layers.Input(shape=(forecast_horizon, 1))
    decoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

## Optimizers & Learning Rate Schedules

### Optimizer Options

```python
# Adam (most common)
optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# RMSprop
optimizer = optimizers.RMSprop(learning_rate=0.001, rho=0.9)

# SGD with momentum
optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# AdaGrad
optimizer = optimizers.Adagrad(learning_rate=0.01)
```

### Learning Rate Schedules

```python
# Exponential decay
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Cosine decay
lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=1000
)
```

## Loss Functions for Forecasting

|Loss Function          |Use Case                 |Implementation                                       |
|-----------------------|-------------------------|-----------------------------------------------------|
|**Mean Squared Error** |General regression       |`loss='mse'` or `tf.keras.losses.MeanSquaredError()` |
|**Mean Absolute Error**|Robust to outliers       |`loss='mae'` or `tf.keras.losses.MeanAbsoluteError()`|
|**Huber Loss**         |Combines MSE & MAE       |`tf.keras.losses.Huber(delta=1.0)`                   |
|**Quantile Loss**      |Probabilistic forecasting|Custom implementation                                |
|**MAPE**               |Percentage-based         |Custom implementation                                |

### Custom Loss Functions

```python
def quantile_loss(quantile):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
    return loss

def mape_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, tf.float32.max))) * 100
```

## Callbacks for Training

```python
# Early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model checkpoint
checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

# Reduce learning rate
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# CSV logger
csv_logger = callbacks.CSVLogger('training_log.csv')

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = callbacks.LearningRateScheduler(scheduler)
```

## Evaluation Metrics

```python
def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive forecasting metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

def directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy"""
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    return np.mean(direction_true == direction_pred)
```

## Complete Forecasting Pipeline

```python
class TimeSeriesForecaster:
    def __init__(self, model_type='lstm', seq_length=60):
        self.model_type = model_type
        self.seq_length = seq_length
        self.model = None
        self.scaler = None
        
    def prepare_data(self, data, target_col=0):
        """Prepare data for training"""
        # Scale data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_sequences(scaled_data, self.seq_length, target_col)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def build_model(self):
        """Build model based on type"""
        input_shape = (self.seq_length, self.X_train.shape[2])
        
        if self.model_type == 'lstm':
            self.model = create_lstm_model(input_shape)
        elif self.model_type == 'gru':
            self.model = create_gru_model(input_shape)
        elif self.model_type == 'cnn_lstm':
            self.model = create_cnn_lstm_model(input_shape)
        
        return self.model
    
    def train(self, epochs=100, batch_size=32):
        """Train the model"""
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def predict(self, data=None):
        """Make predictions"""
        if data is None:
            data = self.X_test
        
        predictions = self.model.predict(data)
        
        # Inverse transform predictions
        predictions_reshaped = predictions.reshape(-1, 1)
        dummy = np.zeros((predictions_reshaped.shape[0], self.scaler.n_features_in_))
        dummy[:, 0] = predictions_reshaped.flatten()
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predictions_original
    
    def forecast_future(self, n_steps, last_sequence):
        """Forecast future values"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_pred = self.model.predict(current_sequence.reshape(1, self.seq_length, -1))
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred
        
        return np.array(predictions)
```

## Advanced Techniques

### Multi-Step Forecasting

```python
def create_multistep_model(input_shape, forecast_horizon):
    """Model for multi-step ahead forecasting"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(forecast_horizon)  # Output multiple time steps
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### Probabilistic Forecasting

```python
def create_probabilistic_model(input_shape):
    """Model that outputs mean and variance"""
    inputs = layers.Input(shape=input_shape)
    lstm = layers.LSTM(64)(inputs)
    
    # Mean output
    mean_output = layers.Dense(1, name='mean')(lstm)
    
    # Variance output (using softplus to ensure positive)
    var_output = layers.Dense(1, activation='softplus', name='variance')(lstm)
    
    model = Model(inputs=inputs, outputs=[mean_output, var_output])
    
    def gaussian_loss(y_true, y_pred):
        mean, var = y_pred[0], y_pred[1]
        return tf.reduce_mean(0.5 * tf.log(var) + 0.5 * tf.square(y_true - mean) / var)
    
    model.compile(optimizer='adam', loss=gaussian_loss)
    return model
```

### Transfer Learning

```python
def fine_tune_pretrained_model(base_model, new_input_shape):
    """Fine-tune a pre-trained model for new data"""
    # Freeze base layers
    for layer in base_model.layers[:-2]:
        layer.trainable = False
    
    # Add new layers
    x = base_model.layers[-3].output
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    
    return model
```

## Model Interpretation & Analysis

### Feature Importance (for CNN models)

```python
def plot_feature_importance(model, X_sample):
    """Plot feature importance using gradients"""
    with tf.GradientTape() as tape:
        tape.watch(X_sample)
        predictions = model(X_sample)
    
    gradients = tape.gradient(predictions, X_sample)
    importance = tf.reduce_mean(tf.abs(gradients), axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(importance.numpy().flatten())
    plt.title('Feature Importance Over Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Importance')
    plt.show()
```

### Residual Analysis

```python
def analyze_residuals(y_true, y_pred):
    """Analyze model residuals"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residual plot
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residual Plot')
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # Residuals over time
    axes[1, 1].plot(residuals)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Over Time')
    
    plt.tight_layout()
    plt.show()
```

## Quick Reference Commands

### Model Summary & Visualization

```python
# Model summary
model.summary()

# Plot model architecture
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# Model configuration
model.get_config()

# Layer weights
model.get_weights()
```

### Training Monitoring

```python
# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.show()
```

### Model Saving & Loading

```python
# Save entire model
model.save('my_forecasting_model.h5')

# Save model architecture only
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)

# Save weights only
model.save_weights('model_weights.h5')

# Load model
model = tf.keras.models.load_model('my_forecasting_model.h5')
```

## Common Hyperparameters

|Parameter          |Range       |Description                  |
|-------------------|------------|-----------------------------|
|**Sequence Length**|10-100      |Length of input sequences    |
|**LSTM/GRU Units** |32-256      |Number of units in RNN layers|
|**Batch Size**     |16-128      |Training batch size          |
|**Learning Rate**  |1e-5 to 1e-2|Optimizer learning rate      |
|**Dropout Rate**   |0.1-0.5     |Regularization dropout rate  |
|**Epochs**         |50-500      |Number of training epochs    |

## Troubleshooting Common Issues

### Overfitting

- Add dropout layers
- Reduce model complexity
- Use early stopping
- Add L1/L2 regularization

### Underfitting

- Increase model complexity
- Add more layers/units
- Reduce regularization
- Train for more epochs

### Vanishing Gradients

- Use LSTM/GRU instead of SimpleRNN
- Apply gradient clipping
- Use batch normalization
- Consider ResNet-style connections

### Poor Convergence

- Adjust learning rate
- Try different optimizers
- Check data preprocessing
- Normalize/standardize inputs


# Pandas Time Series Forecasting Reference Card

## Essential Methods Overview

|Method        |Purpose                             |Returns                |Common Parameters                |
|--------------|------------------------------------|-----------------------|---------------------------------|
|`shift()`     |Move data by n periods              |Series/DataFrame       |`periods`, `freq`, `fill_value`  |
|`diff()`      |Calculate difference between periods|Series/DataFrame       |`periods`                        |
|`pct_change()`|Percentage change between periods   |Series/DataFrame       |`periods`, `fill_method`         |
|`rolling()`   |Create rolling window               |Rolling object         |`window`, `min_periods`, `center`|
|`expanding()` |Cumulative expanding window         |Expanding object       |`min_periods`                    |
|`ewm()`       |Exponential weighted moving         |ExponentialMovingWindow|`span`, `alpha`, `halflife`      |
|`resample()`  |Resample time series data           |Resampler object       |`rule` (e.g., ‘D’, ‘M’, ‘Y’)     |
|`lag()`       |Alias for shift (not built-in)      |Series/DataFrame       |Custom implementation            |

-----

## 1. Shift Operations

### Basic Shifting

```python
import pandas as pd
import numpy as np

# Create sample time series
dates = pd.date_range('2024-01-01', periods=10, freq='D')
df = pd.DataFrame({
    'sales': [100, 120, 115, 130, 125, 140, 135, 150, 145, 160],
    'date': dates
})

# Shift forward (lag) - previous values
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag2'] = df['sales'].shift(2)

# Shift backward (lead) - future values
df['sales_lead1'] = df['sales'].shift(-1)

# Fill NaN values
df['sales_lag1_filled'] = df['sales'].shift(1, fill_value=0)
```

### Creating Multiple Lags

```python
# Create multiple lag features
for i in range(1, 8):
    df[f'lag_{i}'] = df['sales'].shift(i)

# Alternative: Using a loop with dictionary
lags = {f'lag_{i}': df['sales'].shift(i) for i in range(1, 8)}
df = pd.concat([df, pd.DataFrame(lags)], axis=1)
```

-----

## 2. Difference Operations

### First Difference

```python
# Calculate change from previous period
df['sales_diff'] = df['sales'].diff()

# Second difference (difference of differences)
df['sales_diff2'] = df['sales'].diff().diff()

# Percentage change
df['sales_pct_change'] = df['sales'].pct_change()

# Percentage change over n periods
df['sales_pct_change_7d'] = df['sales'].pct_change(periods=7)
```

### Seasonal Difference

```python
# Remove seasonal pattern (e.g., weekly seasonality)
df['sales_seasonal_diff'] = df['sales'].diff(7)

# Remove both trend and seasonality
df['sales_stationary'] = df['sales'].diff().diff(7)
```

-----

## 3. Rolling Window Operations

### Simple Rolling Statistics

```python
# Rolling mean (moving average)
df['ma_7'] = df['sales'].rolling(window=7).mean()
df['ma_30'] = df['sales'].rolling(window=30).mean()

# Rolling sum
df['sum_7'] = df['sales'].rolling(window=7).sum()

# Rolling standard deviation
df['std_7'] = df['sales'].rolling(window=7).std()

# Rolling min/max
df['min_7'] = df['sales'].rolling(window=7).min()
df['max_7'] = df['sales'].rolling(window=7).max()
```

### Advanced Rolling Operations

```python
# Multiple statistics at once
rolling_stats = df['sales'].rolling(window=7).agg(['mean', 'std', 'min', 'max'])

# Custom rolling function
df['range_7'] = df['sales'].rolling(window=7).apply(lambda x: x.max() - x.min())

# Rolling with minimum periods
df['ma_7_min3'] = df['sales'].rolling(window=7, min_periods=3).mean()

# Centered rolling window
df['ma_7_centered'] = df['sales'].rolling(window=7, center=True).mean()
```

### Rolling Correlations and Covariance

```python
# Assuming we have another feature 'temperature'
df['temperature'] = np.random.randn(len(df)) * 10 + 20

# Rolling correlation
df['rolling_corr'] = df['sales'].rolling(window=30).corr(df['temperature'])

# Rolling covariance
df['rolling_cov'] = df['sales'].rolling(window=30).cov(df['temperature'])
```

-----

## 4. Exponentially Weighted Operations

### EWM Basics

```python
# Exponential moving average (EMA)
df['ema_span10'] = df['sales'].ewm(span=10).mean()
df['ema_span30'] = df['sales'].ewm(span=30).mean()

# Using alpha parameter (0 < alpha <= 1)
df['ema_alpha02'] = df['sales'].ewm(alpha=0.2).mean()

# Using halflife
df['ema_halflife7'] = df['sales'].ewm(halflife=7).mean()

# EWM standard deviation
df['ewm_std'] = df['sales'].ewm(span=10).std()
```

### EWM for Volatility

```python
# Returns (percentage change)
returns = df['sales'].pct_change()

# Exponentially weighted volatility
df['volatility'] = returns.ewm(span=20).std()
```

-----

## 5. Expanding Window Operations

### Cumulative Statistics

```python
# Expanding mean (cumulative average)
df['expanding_mean'] = df['sales'].expanding().mean()

# Expanding sum (cumulative sum)
df['cumsum'] = df['sales'].expanding().sum()
# Or simply:
df['cumsum'] = df['sales'].cumsum()

# Expanding min/max
df['expanding_min'] = df['sales'].expanding().min()
df['expanding_max'] = df['sales'].expanding().max()

# Expanding standard deviation
df['expanding_std'] = df['sales'].expanding().std()
```

-----

## 6. Resampling for Forecasting

### Downsampling (Higher to Lower Frequency)

```python
# Daily to weekly
weekly = df.set_index('date').resample('W').agg({
    'sales': ['sum', 'mean', 'max', 'min']
})

# Daily to monthly
monthly = df.set_index('date').resample('M').agg({
    'sales': 'sum'
})

# Custom aggregation
df_resampled = df.set_index('date').resample('W').agg({
    'sales': ['sum', 'mean', lambda x: x.iloc[-1]]  # last value
})
```

### Upsampling (Lower to Higher Frequency)

```python
# Forward fill
upsampled = monthly.resample('D').ffill()

# Backward fill
upsampled = monthly.resample('D').bfill()

# Interpolation
upsampled = monthly.resample('D').interpolate(method='linear')
```

-----

## 7. Advanced Forecasting Features

### Creating Lagged Features Matrix

```python
def create_lagged_features(df, column, lags):
    """Create multiple lagged features"""
    lagged_df = pd.DataFrame(index=df.index)
    for lag in lags:
        lagged_df[f'{column}_lag{lag}'] = df[column].shift(lag)
    return lagged_df

# Usage
lags_to_create = [1, 2, 3, 7, 14, 21, 28]
lagged_features = create_lagged_features(df, 'sales', lags_to_create)
df_modeling = pd.concat([df, lagged_features], axis=1)
```

### Rolling Window Features

```python
def create_rolling_features(df, column, windows):
    """Create rolling statistics for multiple windows"""
    rolling_df = pd.DataFrame(index=df.index)
    for window in windows:
        rolling_df[f'{column}_ma{window}'] = df[column].rolling(window).mean()
        rolling_df[f'{column}_std{window}'] = df[column].rolling(window).std()
        rolling_df[f'{column}_min{window}'] = df[column].rolling(window).min()
        rolling_df[f'{column}_max{window}'] = df[column].rolling(window).max()
    return rolling_df

# Usage
windows_to_create = [7, 14, 30]
rolling_features = create_rolling_features(df, 'sales', windows_to_create)
```

### Date/Time Features

```python
# Extract temporal features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['weekofyear'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
```

### Fourier Features for Seasonality

```python
def add_fourier_features(df, date_col, period, order):
    """Add Fourier terms for capturing seasonality"""
    t = np.arange(len(df))
    for i in range(1, order + 1):
        df[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        df[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)
    return df

# Weekly seasonality (period=7)
df = add_fourier_features(df, 'date', period=7, order=3)
```

-----

## 8. Window Functions for Forecasting

### Rank and Quantile

```python
# Rolling rank
df['rolling_rank'] = df['sales'].rolling(window=30).rank(pct=True)

# Rolling quantile
df['rolling_q25'] = df['sales'].rolling(window=30).quantile(0.25)
df['rolling_median'] = df['sales'].rolling(window=30).quantile(0.50)
df['rolling_q75'] = df['sales'].rolling(window=30).quantile(0.75)
```

### Custom Window Functions

```python
# Rolling skewness
df['rolling_skew'] = df['sales'].rolling(window=30).skew()

# Rolling kurtosis
df['rolling_kurt'] = df['sales'].rolling(window=30).kurt()

# Z-score (rolling standardization)
df['z_score'] = (df['sales'] - df['sales'].rolling(window=30).mean()) / \
                 df['sales'].rolling(window=30).std()
```

-----

## 9. Complete Forecasting Pipeline Example

```python
import pandas as pd
import numpy as np

# Sample data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
np.random.seed(42)
df = pd.DataFrame({
    'date': dates,
    'sales': 100 + np.cumsum(np.random.randn(365)) + 
             10 * np.sin(np.arange(365) * 2 * np.pi / 7)  # weekly pattern
})

# Set date as index
df.set_index('date', inplace=True)

# 1. Create lag features
for i in [1, 7, 14, 28]:
    df[f'sales_lag{i}'] = df['sales'].shift(i)

# 2. Create rolling features
for window in [7, 14, 30]:
    df[f'sales_ma{window}'] = df['sales'].rolling(window).mean()
    df[f'sales_std{window}'] = df['sales'].rolling(window).std()

# 3. Create exponential moving averages
df['sales_ema7'] = df['sales'].ewm(span=7).mean()
df['sales_ema30'] = df['sales'].ewm(span=30).mean()

# 4. Create difference features
df['sales_diff1'] = df['sales'].diff()
df['sales_diff7'] = df['sales'].diff(7)

# 5. Create temporal features
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# 6. Create interaction features
df['ma7_ma30_ratio'] = df['sales_ma7'] / df['sales_ma30']
df['price_position'] = (df['sales'] - df['sales_ma30']) / df['sales_std30']

# 7. Drop NaN values for modeling
df_model = df.dropna()

print(f"Original shape: {df.shape}")
print(f"Model shape: {df_model.shape}")
print(f"\nFeatures created: {df_model.columns.tolist()}")
```

-----

## 10. Best Practices & Tips

### Handling Missing Values

```python
# Forward fill for time series
df['sales_filled'] = df['sales'].fillna(method='ffill')

# Backward fill
df['sales_filled'] = df['sales'].fillna(method='bfill')

# Interpolation
df['sales_filled'] = df['sales'].interpolate(method='linear')
df['sales_filled'] = df['sales'].interpolate(method='time')  # time-weighted
```

### Avoiding Data Leakage

```python
# WRONG - uses future data
df['ma_7'] = df['sales'].rolling(window=7, center=True).mean()

# CORRECT - only uses past data
df['ma_7'] = df['sales'].rolling(window=7).mean()

# When creating features, always shift target variable
df['target'] = df['sales'].shift(-1)  # predict next day
```

### Memory Optimization

```python
# Use appropriate data types
df['sales'] = df['sales'].astype('float32')
df['is_weekend'] = df['is_weekend'].astype('int8')

# Delete intermediate columns
df.drop(['intermediate_col'], axis=1, inplace=True)
```

### Performance Tips

```python
# Use vectorized operations
# SLOW
df['feature'] = df.apply(lambda x: x['sales'] * 2, axis=1)

# FAST
df['feature'] = df['sales'] * 2

# Batch processing for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

-----

## 11. Future Dataset Creation & Target Engineering

### Creating Future Datasets for Forecasting

```python
import pandas as pd
import numpy as np

# Create future dates
last_date = df.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                              periods=30, freq='D')

# Initialize future dataframe
future_df = pd.DataFrame(index=future_dates)

# Add temporal features (known in advance)
future_df['dayofweek'] = future_df.index.dayofweek
future_df['month'] = future_df.index.month
future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
future_df['day'] = future_df.index.day
future_df['quarter'] = future_df.index.quarter

# Add calendar features
future_df['is_month_start'] = future_df.index.is_month_start.astype(int)
future_df['is_month_end'] = future_df.index.is_month_end.astype(int)
future_df['is_quarter_start'] = future_df.index.is_quarter_start.astype(int)
future_df['is_quarter_end'] = future_df.index.is_quarter_end.astype(int)

# Add cyclical encoding for temporal features
future_df['dayofweek_sin'] = np.sin(2 * np.pi * future_df['dayofweek'] / 7)
future_df['dayofweek_cos'] = np.cos(2 * np.pi * future_df['dayofweek'] / 7)
future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
```

### Multi-Step Ahead Target Creation

```python
# Create multiple forecast horizons
for horizon in [1, 7, 14, 30]:
    df[f'target_h{horizon}'] = df['sales'].shift(-horizon)

# Create target using rolling future mean
df['target_next7_mean'] = df['sales'].shift(-1).rolling(window=7).mean()

# Binary classification targets
df['target_increase'] = (df['sales'].shift(-1) > df['sales']).astype(int)
df['target_above_ma'] = (df['sales'].shift(-1) > 
                          df['sales'].rolling(30).mean()).astype(int)

# Regression target with smoothing
df['target_smoothed'] = df['sales'].shift(-1).rolling(3, center=True).mean()
```

### Creating Cumulative Targets

```python
# Cumulative sum for next N days
df['target_cumsum_7d'] = df['sales'][::-1].rolling(7).sum()[::-1].shift(-7)

# Maximum value in next N days
df['target_max_7d'] = df['sales'][::-1].rolling(7).max()[::-1].shift(-7)

# Minimum value in next N days
df['target_min_7d'] = df['sales'][::-1].rolling(7).min()[::-1].shift(-7)
```

-----

## 12. Advanced Feature Engineering Strategies

### Interaction Features

```python
# Multiplicative interactions
df['ma7_x_dow'] = df['sales_ma7'] * df['dayofweek']
df['lag1_x_weekend'] = df['sales_lag1'] * df['is_weekend']

# Ratio features
df['sales_to_ma7'] = df['sales'] / df['sales_ma7']
df['sales_to_ma30'] = df['sales'] / df['sales_ma30']
df['ma7_to_ma30'] = df['sales_ma7'] / df['sales_ma30']

# Difference features
df['diff_from_ma7'] = df['sales'] - df['sales_ma7']
df['diff_from_ma30'] = df['sales'] - df['sales_ma30']
```

### Statistical Features

```python
# Rolling z-score
df['zscore_30'] = (df['sales'] - df['sales'].rolling(30).mean()) / \
                   df['sales'].rolling(30).std()

# Rolling coefficient of variation
df['cv_30'] = df['sales'].rolling(30).std() / df['sales'].rolling(30).mean()

# Rolling range
df['range_30'] = df['sales'].rolling(30).max() - df['sales'].rolling(30).min()

# Percentile rank
df['percentile_rank_30'] = df['sales'].rolling(30).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
)
```

### Momentum and Velocity Features

```python
# Rate of change
df['roc_7'] = (df['sales'] - df['sales'].shift(7)) / df['sales'].shift(7)

# Acceleration (second derivative)
df['acceleration'] = df['sales'].diff().diff()

# Momentum
df['momentum_7'] = df['sales'] - df['sales'].shift(7)
df['momentum_14'] = df['sales'] - df['sales'].shift(14)

# Moving average crossover signals
df['ma_crossover'] = (df['sales_ma7'] > df['sales_ma30']).astype(int)
```

### Volatility Features

```python
# Historical volatility
returns = df['sales'].pct_change()
df['volatility_7'] = returns.rolling(7).std()
df['volatility_30'] = returns.rolling(30).std()

# Parkinson volatility (using high-low)
# Requires high and low data
# df['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * 
#                        np.log(df['high']/df['low'])**2)

# Average True Range (ATR) approximation
df['atr_14'] = df['sales'].diff().abs().rolling(14).mean()
```

-----

## 13. Autocorrelation Analysis

### Computing Autocorrelation

```python
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

# Autocorrelation function
acf_values = [df['sales'].autocorr(lag=i) for i in range(1, 31)]

# Partial autocorrelation (requires statsmodels)
from statsmodels.tsa.stattools import acf, pacf

# ACF values
acf_vals = acf(df['sales'].dropna(), nlags=30)
# PACF values
pacf_vals = pacf(df['sales'].dropna(), nlags=30)

# Plot autocorrelation
autocorrelation_plot(df['sales'])
plt.show()
```

### Creating Autocorrelation Features

```python
# Significant lag correlations as features
significant_lags = [1, 7, 14, 28]  # Based on ACF analysis
for lag in significant_lags:
    df[f'sales_lag{lag}'] = df['sales'].shift(lag)

# Rolling autocorrelation
df['rolling_acf1'] = df['sales'].rolling(30).apply(
    lambda x: x.autocorr(lag=1), raw=False
)
```

-----

## 14. Stationarity Analysis & Transformation

### Testing for Stationarity

```python
from statsmodels.tsa.stattools import adfuller, kpss

# Augmented Dickey-Fuller test
def adf_test(series, name=''):
    result = adfuller(series.dropna())
    print(f'ADF Test for {name}')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    print(f'Stationary: {"Yes" if result[1] < 0.05 else "No"}\n')

# KPSS test (null hypothesis: series is stationary)
def kpss_test(series, name=''):
    result = kpss(series.dropna(), regression='ct')
    print(f'KPSS Test for {name}')
    print(f'KPSS Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Stationary: {"Yes" if result[1] > 0.05 else "No"}\n')

# Run tests
adf_test(df['sales'], 'Sales')
kpss_test(df['sales'], 'Sales')
```

### Making Data Stationary

```python
# First differencing
df['sales_diff1'] = df['sales'].diff()

# Second differencing
df['sales_diff2'] = df['sales'].diff().diff()

# Log transformation (for stabilizing variance)
df['sales_log'] = np.log(df['sales'])
df['sales_log_diff'] = np.log(df['sales']).diff()

# Box-Cox transformation
from scipy.stats import boxcox
df['sales_boxcox'], lambda_param = boxcox(df['sales'])

# Seasonal differencing
df['sales_seasonal_diff'] = df['sales'].diff(7)  # weekly

# Combined trend and seasonal differencing
df['sales_stationary'] = df['sales'].diff().diff(7)

# Detrending using rolling mean
df['sales_detrended'] = df['sales'] - df['sales'].rolling(30).mean()

# Percentage change (removes trend and scale)
df['sales_pct'] = df['sales'].pct_change()
```

-----

## 15. Seasonality & Cyclicity Analysis

### Detecting Seasonality

```python
# Simple seasonal subseries plots
def seasonal_plot(df, column, freq='M'):
    df_temp = df.copy()
    df_temp['year'] = df_temp.index.year
    df_temp['period'] = df_temp.index.month if freq == 'M' else df_temp.index.dayofweek
    
    pivot = df_temp.pivot_table(values=column, 
                                 index='period', 
                                 columns='year', 
                                 aggfunc='mean')
    pivot.plot(marker='o', figsize=(12, 6))
    plt.title(f'Seasonal Plot - {column}')
    plt.show()

# Seasonal strength metric
def seasonal_strength(series, period):
    detrended = series - series.rolling(period, center=True).mean()
    seasonal_component = detrended.groupby(
        detrended.index.dayofyear if period <= 365 else detrended.index.month
    ).transform('mean')
    remainder = detrended - seasonal_component
    
    var_seasonal = seasonal_component.var()
    var_remainder = remainder.var()
    
    strength = max(0, 1 - var_remainder / (var_seasonal + var_remainder))
    return strength

# Calculate seasonal strength
strength = seasonal_strength(df['sales'], period=7)
print(f'Seasonal Strength: {strength:.4f}')
```

### Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Classical decomposition (additive)
decomposition = seasonal_decompose(df['sales'], 
                                   model='additive', 
                                   period=7,
                                   extrapolate_trend='freq')

# Extract components
df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
df['sales'].plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Multiplicative decomposition (for exponential trends)
decomposition_mult = seasonal_decompose(df['sales'], 
                                        model='multiplicative', 
                                        period=7,
                                        extrapolate_trend='freq')
```

### STL Decomposition (Seasonal-Trend decomposition using LOESS)

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition (more robust)
stl = STL(df['sales'], seasonal=7, trend=15)
result = stl.fit()

df['stl_trend'] = result.trend
df['stl_seasonal'] = result.seasonal
df['stl_residual'] = result.resid

# Plot STL decomposition
result.plot()
plt.show()
```

### Creating Seasonal Features

```python
# Using decomposition components as features
df['seasonal_component'] = decomposition.seasonal
df['trend_component'] = decomposition.trend
df['deseasonalized'] = df['sales'] - df['seasonal_component']

# Seasonal indices
seasonal_indices = df.groupby(df.index.dayofweek)['sales'].mean()
df['seasonal_index'] = df.index.dayofweek.map(seasonal_indices)

# Seasonal adjustment
overall_mean = df['sales'].mean()
df['seasonal_adjusted'] = df['sales'] / (df['seasonal_index'] / overall_mean)
```

### Fourier Terms for Multiple Seasonalities

```python
def add_fourier_terms(df, periods, orders):
    """
    Add Fourier terms for multiple seasonal periods
    periods: list of seasonal periods (e.g., [7, 365.25])
    orders: list of Fourier orders for each period
    """
    t = np.arange(len(df))
    
    for period, order in zip(periods, orders):
        for i in range(1, order + 1):
            df[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
            df[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)
    
    return df

# Add weekly and yearly seasonality
df = add_fourier_terms(df, periods=[7, 365.25], orders=[3, 5])
```

-----

## 16. Advanced Temporal Resampling

### Resampling with Custom Aggregations

```python
# Multiple aggregation functions
df_daily = df.resample('D').agg({
    'sales': ['sum', 'mean', 'std', 'min', 'max', 'median'],
    'temperature': ['mean', 'min', 'max']
})

# Custom aggregation functions
def custom_agg(x):
    return pd.Series({
        'total': x.sum(),
        'average': x.mean(),
        'volatility': x.std(),
        'range': x.max() - x.min(),
        'last': x.iloc[-1] if len(x) > 0 else np.nan,
        'first': x.iloc[0] if len(x) > 0 else np.nan
    })

df_weekly = df.resample('W').apply(custom_agg)
```

### Resampling with OHLC (Open-High-Low-Close)

```python
# Financial-style aggregation
ohlc = df['sales'].resample('W').ohlc()
print(ohlc.head())

# Volume-weighted average
df_temp = df.copy()
df_temp['volume'] = np.random.randint(100, 1000, len(df))
df_temp['vwap'] = (df_temp['sales'] * df_temp['volume']).resample('W').sum() / \
                   df_temp['volume'].resample('W').sum()
```

### Rolling Resampling

```python
# Combine rolling and resampling
df_hourly = df.resample('H').mean()  # Upsample to hourly
df_hourly['ma_24h'] = df_hourly['sales'].rolling(24).mean()

# Resample with forward-looking window
df_weekly = df.resample('W').agg({
    'sales': 'sum',
    'sales_next_week': lambda x: x.iloc[0] if len(x) > 0 else np.nan
})
```

-----

## 17. Temporal Feature Engineering Patterns

### Cyclical Encoding

```python
# Sine-cosine encoding for circular features
def encode_cyclical(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

df = encode_cyclical(df, 'month', 12)
df = encode_cyclical(df, 'dayofweek', 7)
df = encode_cyclical(df, 'hour', 24)  # if hourly data
```

### Special Periods and Events

```python
# Holidays (example with US holidays)
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df.index.min(), end=df.index.max())

df['is_holiday'] = df.index.isin(holidays).astype(int)
df['days_since_holiday'] = 0
df['days_until_holiday'] = 0

for idx in df.index:
    past_holidays = holidays[holidays < idx]
    future_holidays = holidays[holidays > idx]
    
    if len(past_holidays) > 0:
        df.loc[idx, 'days_since_holiday'] = (idx - past_holidays[-1]).days
    
    if len(future_holidays) > 0:
        df.loc[idx, 'days_until_holiday'] = (future_holidays[0] - idx).days

# Pay period features (bi-weekly)
df['is_payday'] = ((df.index.day == 15) | (df.index.day == 30)).astype(int)

# Week of month
df['week_of_month'] = (df.index.day - 1) // 7 + 1
```

### Lag Interaction Features

```python
# Lag * temporal features
df['lag1_x_dow'] = df['sales_lag1'] * df['dayofweek']
df['lag7_x_month'] = df['sales_lag7'] * df['month']

# Rolling mean * temporal
df['ma7_x_is_weekend'] = df['sales_ma7'] * df['is_weekend']

# Conditional lags
df['lag1_if_weekday'] = df['sales_lag1'] * (1 - df['is_weekend'])
df['lag1_if_weekend'] = df['sales_lag1'] * df['is_weekend']
```

-----

## 18. Window-Based Forecasting Features

### Expanding Window Statistics by Group

```python
# Expanding mean by day of week
df['expanding_mean_by_dow'] = df.groupby('dayofweek')['sales'].expanding().mean().reset_index(0, drop=True)

# Expanding std by month
df['expanding_std_by_month'] = df.groupby('month')['sales'].expanding().std().reset_index(0, drop=True)
```

### Custom Rolling Functions

```python
# Rolling linear trend
def rolling_trend(x):
    if len(x) < 2:
        return np.nan
    y = np.arange(len(x))
    coeffs = np.polyfit(y, x, 1)
    return coeffs[0]  # slope

df['trend_7'] = df['sales'].rolling(7).apply(rolling_trend, raw=True)
df['trend_30'] = df['sales'].rolling(30).apply(rolling_trend, raw=True)

# Rolling peak detection
def count_peaks(x):
    if len(x) < 3:
        return 0
    return np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))

df['peaks_30'] = df['sales'].rolling(30).apply(count_peaks, raw=True)
```

-----

## 19. Cross-Validation for Time Series

### Time Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

# Create splits
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    print(f'Train: {train.index.min()} to {train.index.max()}')
    print(f'Test: {test.index.min()} to {test.index.max()}\n')
```

### Walk-Forward Validation

```python
# Manual walk-forward
def walk_forward_validation(df, initial_train_size, step_size, horizon):
    results = []
    
    for i in range(initial_train_size, len(df) - horizon, step_size):
        train = df.iloc[:i]
        test = df.iloc[i:i+horizon]
        
        # Train model here
        # Make predictions
        # Evaluate
        
        results.append({
            'train_end': train.index[-1],
            'test_start': test.index[0],
            'test_end': test.index[-1]
        })
    
    return pd.DataFrame(results)

validation_splits = walk_forward_validation(df, 
                                            initial_train_size=200,
                                            step_size=30,
                                            horizon=7)
```

-----

## Common Frequency Aliases

|Alias |Description  |
|------|-------------|
|D     |Calendar day |
|B     |Business day |
|W     |Weekly       |
|M     |Month end    |
|MS    |Month start  |
|Q     |Quarter end  |
|QS    |Quarter start|
|Y     |Year end     |
|YS    |Year start   |
|H     |Hourly       |
|T, min|Minutely     |
|S     |Secondly     |

-----

## Quick Reference: Method Chaining

```python
# Efficient method chaining for feature engineering
forecast_df = (df
    .assign(
        lag1 = lambda x: x['sales'].shift(1),
        lag7 = lambda x: x['sales'].shift(7),
        ma7 = lambda x: x['sales'].rolling(7).mean(),
        ma30 = lambda x: x['sales'].rolling(30).mean(),
        ema7 = lambda x: x['sales'].ewm(span=7).mean(),
        diff1 = lambda x: x['sales'].diff(),
        pct_change = lambda x: x['sales'].pct_change()
    )
    .dropna()
)
```

