import pandas as pd
import datetime as dt
import numpy as np
import cloudpickle
import joblib as jl

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import initializers

import statsmodels.tsa as tsa
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import ensemble
import sklearn.preprocessing as prep

############################################################################################################################################

"""
Parameters:
    data: pd.DataFrame with one column (for prices)
    lag: int
        - Number of days to go back in order to make forecast, e.g. day_lag=7 indicates a weekly persistent model,
        day_lag=1 indicates a daily persistent model, etc.
"""
class naive():
    def __init__(self, lag, period):
        self.lag = lag
        self.period = period
    
    def ingest_data(self, train_target, train_bm, train_planned):
        self.data = train_target.copy()
       
    def train(self):
        # Date for which forecast will be made
        self.target_date = dt.datetime.combine(self.data.index.date[-1], dt.datetime.min.time()) + dt.timedelta(days=1)
    
    def forecast(self):
        # Make forecasts
        if self.period == "D":
            forecast_df = self.data.loc[self.data.index.date == self.data.index.date[-(self.lag*24)]]
        elif self.period == "Y":
            # Get corresponding date from lag years before    
            try:
                forecast_date = dt.datetime(self.target_date.year-1, self.target_date.month, self.target_date.day)
            except:
                forecast_date = dt.datetime(self.target_date.year-1, self.target_date.month, self.target_date.day-1)
            
            forecast_df = self.data.loc[self.data.index.date == forecast_date.date()]

        # Create and prepare forecasts dataframe
        forecast_df.index = pd.date_range(self.target_date, periods=24, freq="H")
        forecast_df.index.name = "DeliveryPeriod"

        return(forecast_df)

############################################################################################################################################
    
"""
Parameters:
    model_params: dict with keys: n_estimators, max_depth, max_features, n_jobs
    lag_params: dict with keys: price_lags, bm_price_lags, planned_lags
"""
class random_forest():
    def __init__(self, model_params, lag_params):
        self.model = ensemble.RandomForestRegressor(**model_params, oob_score=True)
        self.price_lags = lag_params["price_lags"]
        self.bm_price_lags = lag_params["bm_price_lags"]
        self.planned_lags = lag_params["planned_lags"]
        
    def ingest_data(self, train_target, train_bm, train_planned):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq="h")
    
        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)
        
        # Build predictors from EURPrices
        for lag in self.price_lags:
            predictor_name = f"{train_target.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_target)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)
        
        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]
        
        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]
        
        # Store data into object
        self.train_predictors = train_predictors
        self.test_predictors = test_predictors
        self.train_target = train_target
        
        # Initialise dataframe for variable importances
        if not hasattr(self, "variable_importances"):
            self.variable_importances = pd.DataFrame(index=train_predictors.columns)
        
    def train(self):
        # Fit the forecasting model
        self.model.fit(X=self.train_predictors, y=self.train_target["EURPrices"])
        
        # Store variable importances
        variable_importances = self.model.feature_importances_
        self.variable_importances.insert(self.variable_importances.shape[1], self.test_predictors.index.date[0], variable_importances)

    def forecast(self):
        forecast_df = pd.DataFrame(self.model.predict(self.test_predictors), index=self.test_predictors.index)
        forecast_df.index.name = "DeliveryPeriod"
        return(forecast_df)

############################################################################################################################################
    
"""
Parameters:
    model_params: dict with keys: lags, trend, ic, exog
    lag_params: dict with keys: bm_price_lags, planned_lags
"""
class ARX():
    def __init__(self, model_params, lag_params):
        self.lags = model_params["lags"]
        self.trend = model_params["trend"]
        self.ic = model_params["ic"]
        self.exog = model_params["exog"]
        
        self.bm_price_lags = lag_params["bm_price_lags"]
        self.planned_lags = lag_params["planned_lags"]
        
    def ingest_data(self, train_target, train_bm, train_planned):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq="h")
    
        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)
        
        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]
        
        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]
        
        # Store data into object
        self.train_predictors = train_predictors
        self.test_predictors = test_predictors
        self.train_target = train_target
        
        # Initialise dataframe for variable importances
#         if not hasattr(self, "variable_importances"):
#             self.variable_importances = pd.DataFrame(index=train_predictors.columns)
        
    def train(self):
        # Fit AR/ARX models with different lag values
        if self.exog:
            model_selection = jl.Parallel(n_jobs=-1, backend="threading") \
                (jl.delayed(tsa.ar_model.AutoReg(self.train_target.values,lag,self.trend,exog=self.train_predictors.values).fit)() for lag in self.lags)
        else:
            model_selection = jl.Parallel(n_jobs=-1, backend="threading") \
                (jl.delayed(tsa.ar_model.AutoReg(self.train_target.values,lag,self.trend).fit)() for lag in self.lags)
        
        # Pick best model based on the specified information criterion (AIC or BIC)
        ar_ics = [model.aic for model in model_selection] if self.ic=='aic' else [model.bic for model in model_selection]
        self.model = model_selection[ar_ics.index(min(ar_ics))]
        
    def forecast(self):
        # Make forecasts
        if self.exog:
            forecast = self.model.predict(start=self.train_target.shape[0], end=self.train_target.shape[0]+23, exog_oos=self.test_predictors)
        else:
            forecast = self.model.predict(start=self.train_target.shape[0], end=self.train_target.shape[0]+23)
        
        # Store forecasts in labelled dataframe
        forecast_df = pd.DataFrame(dict(Forecast=forecast), index=self.test_predictors.index)
        return(forecast_df)
    
############################################################################################################################################
    
"""
Parameters:
    model_params: dict with keys: exog, trend, order, seasonal_order, method, maxiter, disp
    lag_params: dict with keys: bm_price_lags, planned_lags
"""
class SARIMAX():
    def __init__(self, model_params, lag_params):
        self.exog = model_params["exog"]
        self.trend = model_params["trend"]
        self.order = model_params["order"]
        self.seasonal_order = model_params["seasonal_order"]
        self.method = model_params["method"]
        self.maxiter = model_params["maxiter"]
        self.disp = model_params["disp"]
        
        self.bm_price_lags = lag_params["bm_price_lags"]
        self.planned_lags = lag_params["planned_lags"]
        
    def ingest_data(self, train_target, train_bm, train_planned):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq="h")
    
        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)
        
        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]
        
        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]
        
        # Store data into object
        self.train_predictors = train_predictors
        self.test_predictors = test_predictors
        self.train_target = train_target
        
        # Initialise dataframe for variable importances
#         if not hasattr(self, "variable_importances"):
#             self.variable_importances = pd.DataFrame(index=train_predictors.columns)
        
    def train(self):
        # Fit AR/ARX models with different lag values
        if self.exog:
            self.model = tsa.statespace.sarimax.SARIMAX(endog=self.train_target,exog=self.train_predictors,
                                                        order=self.order,seasona_order=self.seasonal_order,
                                                        trend=self.trend).fit(method=self.method,maxiter=self.maxiter,disp=self.disp)
        else:
            self.model = tsa.statespace.sarimax.SARIMAX(endog=self.train_target,order=self.order,seasonal_order=self.seasonal_order,
                                                        trend=self.trend).fit(method=self.method,maxiter=self.maxiter,disp=self.disp)
        
    def forecast(self):
        # Make forecasts
        if self.exog:
            forecast = self.model.predict(start=self.train_predictors.shape[0], end=self.train_predictors.shape[0]+23, exog=self.test_predictors)
        else:
            forecast = self.model.predict(start=self.train_predictors.shape[0], end=self.train_predictors.shape[0]+23)
        forecast_df = pd.DataFrame(dict(Forecast=forecast), index=self.test_predictors.index)
        return(forecast_df)
    
############################################################################################################################################

def create_ffnn(num_of_nodes, input_cols, act_fn, n_layers=3, opt="adam", loss="mse"):
    model = Sequential()
    initializer = initializers.he_uniform(1)
    
    # Add first hidden layer (with input layer specification)
    model.add(Dense(num_of_nodes, activation=act_fn, input_shape=(input_cols,), kernel_initializer=initializer))
    
    # Add remaining hidden layers
    for _ in range(n_layers-1):
        model.add(Dense(num_of_nodes, activation=act_fn, kernel_initializer=initializer))

    # Add output layer
    model.add(Dense(1, kernel_initializer=initializer))

#     optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss=loss)

    return(model)


def scale_predictors(predictors, activation, copy_df=True):
    activation_ranges = {
        "tanh": (-1,1),
        "sigmoid": (0,1),
        "relu": (0,5)
    }
    scaler = prep.MinMaxScaler(feature_range=activation_ranges[activation], copy=copy_df)
    scaled_predictors = pd.DataFrame(scaler.fit_transform(predictors), index=predictors.index, columns=predictors.columns)
    
    return(scaler, scaled_predictors)


"""
Parameters:
    model_params: dict with keys: init_params, train_params, other_params
    init_params: dict for __init__ model params
    train_params: dict for model.fit() params
    other_params: dict for other params that we need to specify in a unique way
    lag_params: dict with keys: price_lags, bm_price_lags, planned_lags
"""
class ffnn():
    def __init__(self, model_params, lag_params):
        self.init_params = model_params["init_params"]
        self.train_params = model_params["train_params"]
        self.other_params = model_params["other_params"]
        
        if not hasattr(self, "model"):
            self.model = create_ffnn(**self.init_params)
        
        self.price_lags = lag_params["price_lags"]
        self.bm_price_lags = lag_params["bm_price_lags"]
        self.planned_lags = lag_params["planned_lags"]
        
    def ingest_data(self, train_target, train_bm, train_planned):
        start_dates = [df.index[0] for df in [train_target, train_bm, train_planned]]
        latest_start_date = max(start_dates)

        # Split into training predictors and test predictors
        last_day_of_data = dt.datetime.combine(train_planned.index.date[-1], dt.datetime.min.time())
        test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq="h")

        # Initialise predictors dataframe
        predictors = pd.DataFrame(index=train_planned.index)

        # Build predictors from EURPrices
        for lag in self.price_lags:
            predictor_name = f"{train_target.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_target)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)

        # Build predictors from ImbalancePrice
        for lag in self.bm_price_lags:
            predictor_name = f"{train_bm.columns[0]}-{lag}"
            predictors.insert(predictors.shape[1], predictor_name, train_bm)
            predictors[predictor_name] = predictors[predictor_name].shift(lag)
            
        # Build predictors from Wind and Demand
        for column in train_planned:
            for lag in self.planned_lags:
                predictor_name = f"{column}-{lag}"
                predictors.insert(predictors.shape[1], predictor_name, train_planned[column])
                predictors[predictor_name] = predictors[predictor_name].shift(lag)
                
        # Split predictors into training and test predictors and store data for training and forecasting
        train_predictors = predictors.drop(test_index)
        test_predictors = predictors.loc[test_index,:]

        # Remove rows in training set with NAs
        notna_train_predictors_loc = train_predictors.notna().all(axis=1)
        train_predictors = train_predictors.loc[notna_train_predictors_loc]
        train_target = train_target.loc[notna_train_predictors_loc]

        # Scale predictors and target, and store in object
        self.train_predictors_scaler, self.scaled_train_predictors = scale_predictors(train_predictors, activation=self.init_params["act_fn"])
        self.train_target_scaler, self.scaled_train_target = scale_predictors(train_target, activation=self.init_params["act_fn"])
        self.scaled_test_predictors = self.train_predictors_scaler.transform(test_predictors)
        
        self.test_index = test_predictors.index
        
        # Initialise dataframe for variable importances
        if not hasattr(self, "variable_importances"):
            self.variable_importances = pd.DataFrame(index=train_predictors.columns)

    def train(self):
        # Fit the forecasting model
#         self.model.fit(x=self.scaled_train_predictors, y=self.scaled_train_target, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
        if hasattr(self, "history"):
            self.history = self.model.fit(x=self.scaled_train_predictors, y=self.scaled_train_target, epochs=self.other_params["subseq_epochs"], batch_size=len(self.scaled_train_target), **self.train_params)
        else:
            self.history = self.model.fit(x=self.scaled_train_predictors, y=self.scaled_train_target, epochs=self.other_params["init_epochs"], batch_size=len(self.scaled_train_target), **self.train_params)
        
#         # Store variable importances
#         variable_importances = self.model.feature_importances_
#         self.variable_importances.insert(self.variable_importances.shape[1], self.test_predictors.index.date[0], variable_importances)

    def forecast(self):
        forecast = self.train_target_scaler.inverse_transform(self.model.predict(x=self.scaled_test_predictors))
        forecast_df = pd.DataFrame(forecast, index=self.test_index)
        return(forecast_df)
    
############################################################################################################################################

def create_data_window(data, n_steps, step_size=24):
    new_data = data.copy()
    column = data.columns[0]

    # Add lagged values as new columns
    for step in range(n_steps):
        new_data.insert(new_data.shape[1], f"{column}-{step_size*(step+1)}h", new_data[[column]].shift(step_size * (step+1)))
        
    # Remove rows with NAs
    new_data = new_data.loc[new_data.notna().all(axis=1)]
    
    return(new_data)


def get_rnn_scaler(predictors, activation, copy_df=True):
    activation_ranges = {
        "tanh": (-1,1),
        "sigmoid": (0,1),
        "relu": (0,5)
    }
    real_predictors = predictors[:,0,:].copy()
    
    # Create and fit scaler
    scaler = prep.MinMaxScaler(feature_range=activation_ranges[activation], copy=copy_df)
    scaler.fit(real_predictors)
    
    return(scaler)


def transform_rnn_scale(predictors, scaler):
    # Transform predictors
    scaled_predictors = predictors.copy()
    
    for time_step in range(scaled_predictors.shape[1]):
        scaled_predictors[:,time_step,:] = scaler.transform(scaled_predictors[:,time_step,:])
        
    return(scaled_predictors)


def invert_rnn_scale(scaled_predictors, scaler):
    original_predictors = scaled_predictors.copy()
    
    for time_step in range(original_predictors.shape[1]):
        original_predictors[:,time_step,:] = scaler.inverse_transform(scaled_predictors[:,time_step,:])
        
    return(original_predictors)


def create_rnn(num_of_blocks, n_timesteps, n_features, act_fn, n_layers=1, opt="adam", loss="mse"):
    model = Sequential()
#     initializer = initializers.he_uniform(1)

    # Add first hidden layer
    if n_layers == 1:
        model.add(LSTM(num_of_blocks, activation=act_fn, input_shape=(n_timesteps, n_features)))#, batch_input_shape=(24, n_timesteps, n_features), stateful=True))#, kernel_initializer=initializer))
    else:
        model.add(LSTM(num_of_blocks, return_sequences=True, activation=act_fn, input_shape=(n_timesteps, n_features)))#, batch_input_shape=(24, n_timesteps, n_features), stateful=True))#, kernel_initializer=initializer))
    
    # Add remaining hidden layers
    for n in range(n_layers-1):
        if n == n_layers-2:
            model.add(LSTM(num_of_blocks, activation=act_fn))#, batch_input_shape=(24, n_timesteps, n_features), stateful=True))#, kernel_initializer=initializer))
        else:
            model.add(LSTM(num_of_blocks, return_sequences=True, activation=act_fn))#, batch_input_shape=(24, n_timesteps, n_features), stateful=True))#, kernel_initializer=initializer))

    # Add output layer
    model.add(Dense(1))#, kernel_initializer=initializer))

#     optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss=loss)

    return(model)


"""
Parameters:
    model_params: dict with keys: init_params, train_params, other_params
    init_params: dict for __init__ model params
    train_params: dict for model.fit() params
    other_params: dict for other params that we need to specify in a unique way
"""

class rnn():
    def __init__(self, model_params, lag_params):
        self.init_params = model_params["init_params"]
        self.train_params = model_params["train_params"]
        self.other_params = model_params["other_params"]
        
        if not hasattr(self, "model"):
            self.model = create_rnn(**self.init_params)
        
    def ingest_data(self, train_target, train_bm, train_planned):
        # Get the the latest start date of the variables
        start_dates = list(map(lambda var: var.index[0], [train_target, train_bm, train_planned]))
        latest_start_date = max(start_dates)
        target_end_date = train_target.index[-1]
        
        # Match data start dates
        train_target = train_target.loc[train_target.index.date >= latest_start_date.date()]
        train_bm = train_bm.loc[train_bm.index.date >= latest_start_date.date()]
        train_planned = train_planned.loc[train_planned.index.date >= latest_start_date.date()]
        
        # Match data end dates
        train_target = train_target.loc[train_target.index.date <= train_target.index.date[-1]]
        train_bm = train_bm.loc[train_bm.index.date <= train_target.index.date[-1]]
        train_planned = train_planned.loc[train_planned.index.date <= train_target.index.date[-1]]
        
        # Split up planned data into its component columns
        train_wind = train_planned.loc[:,"Wind"].to_frame()
        train_demand = train_planned.loc[:,"Demand"].to_frame()

        # Reformat data for input into RNNs
        n_timesteps = self.init_params["n_timesteps"]
        rnn_train_prices = create_data_window(train_target, n_timesteps)
        rnn_train_target = rnn_train_prices[[rnn_train_prices.columns[0]]]
        rnn_train_prices.drop(rnn_train_prices.columns[0], axis=1, inplace=True)

        rnn_train_bm = create_data_window(train_bm, n_timesteps)
        rnn_train_bm.drop(rnn_train_bm.columns[0], axis=1, inplace=True)

        rnn_train_wind = create_data_window(train_wind, n_timesteps)
        rnn_train_wind.drop(rnn_train_wind.columns[-1], axis=1, inplace=True)

        rnn_train_demand = create_data_window(train_demand, n_timesteps)
        rnn_train_demand.drop(rnn_train_demand.columns[-1], axis=1, inplace=True)

        # Split into training and test set
        last_day_of_data = dt.datetime.combine(train_target.index.date[-1], dt.datetime.min.time())
        self.test_index = pd.date_range(start=last_day_of_data, end=last_day_of_data+dt.timedelta(hours=23), freq="H")

        # Test data
        rnn_test_prices = rnn_train_prices.loc[self.test_index]
        rnn_test_bm = rnn_train_bm.loc[self.test_index]
        rnn_test_wind = rnn_train_wind.loc[self.test_index]
        rnn_test_demand = rnn_train_demand.loc[self.test_index]

        # Train data
        rnn_train_prices = rnn_train_prices.loc[:last_day_of_data-dt.timedelta(hours=1)]
        rnn_train_bm = rnn_train_bm.loc[:last_day_of_data-dt.timedelta(hours=1)]
        rnn_train_wind = rnn_train_wind.loc[:last_day_of_data-dt.timedelta(hours=1)]
        rnn_train_demand = rnn_train_demand.loc[:last_day_of_data-dt.timedelta(hours=1)]
        rnn_train_target = rnn_train_target.loc[:last_day_of_data-dt.timedelta(hours=1)]

        # Combine train and test features into one tensor each
        rnn_test_predictors = np.hstack((rnn_test_prices, rnn_test_bm, rnn_test_wind, rnn_test_demand)).reshape(rnn_test_prices.shape[0], 4, n_timesteps).transpose(0,2,1)
        rnn_train_predictors = np.hstack((rnn_train_prices, rnn_train_bm, rnn_train_wind, rnn_train_demand)).reshape(rnn_train_prices.shape[0], 4, n_timesteps).transpose(0,2,1)

        # Scale predictors and target, and store in object
        self.rnn_train_predictors_scaler = get_rnn_scaler(rnn_train_predictors, activation=self.init_params["act_fn"])
        self.rnn_scaled_train_predictors = transform_rnn_scale(rnn_train_predictors, self.rnn_train_predictors_scaler)
        self.rnn_scaled_test_predictors = transform_rnn_scale(rnn_test_predictors, self.rnn_train_predictors_scaler)
        self.rnn_train_target_scaler, self.rnn_scaled_train_target = scale_predictors(rnn_train_target, activation=self.init_params["act_fn"])
        
#         if not hasattr(self, "feature_names"):
#             self.feature_names = train_predictors.columns.tolist()
        
#         # Initialise dataframe for variable importances
#         if not hasattr(self, "variable_importances"):
#             self.variable_importances = pd.DataFrame(index=train_predictors.columns)

    def train(self):
        # Fit the forecasting model
        if hasattr(self, "history"):
            self.history = self.model.fit(x=self.rnn_scaled_train_predictors, y=self.rnn_scaled_train_target.values, epochs=self.other_params["subseq_epochs"], **self.train_params)
        else:
            self.history = self.model.fit(x=self.rnn_scaled_train_predictors, y=self.rnn_scaled_train_target.values, epochs=self.other_params["init_epochs"], **self.train_params)
        
#         # Store variable importances
#         variable_importances = self.model.feature_importances_
#         self.variable_importances.insert(self.variable_importances.shape[1], self.test_predictors.index.date[0], variable_importances)

    def forecast(self):
        forecast = self.rnn_train_target_scaler.inverse_transform(self.model.predict(x=self.rnn_scaled_test_predictors))
        forecast_df = pd.DataFrame(forecast, index=self.test_index+dt.timedelta(days=1))
        print()
        return(forecast_df)