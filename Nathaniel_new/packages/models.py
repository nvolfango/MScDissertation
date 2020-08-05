import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.tsa as tsa
import statsmodels.api as sm
import statsmodels.formula.api as smf

"""
Parameters:
    data: pd.DataFrame
    day_lag: int
        - Number of days to go back in order to make forecast, e.g. day_lag=7 indicates a weekly persistent model,
        day_lag=1 indicates a daily persistent model, etc.
"""

def naive(data, day_lag):
    # Make forecasts
    forecast_df = data.loc[data.index.date == data.index.date[-(day_lag*24)]]
        
    # Create and prepare forecasts dataframe
    forecast_df.index = pd.date_range(start=dt.datetime.combine(data.index.date[-1], dt.datetime.min.time())+dt.timedelta(days=1), periods=24, freq="H")
    forecast_df.index.name = "DeliveryPeriod"
    
    return(forecast_df)


"""
Parameters:
    data: pd.DataFrame
    target: datetime
    p: int
        AR model parameter
    trend: str
    forecast_steps: int
"""

def AR_univariate(data, target, p, trend='ct', forecast_steps=24):
    
    if type(target) == str:
        target = dt.datetime.strptime(target, "%Y-%m-%d")
    
    # Fit AR(p) model to data
    ar_model = tsa.ar_model.AutoReg(endog=data, lags=p, trend=trend).fit()
    
    # Make forecasts
    forecast = ar_model.predict(start=target, end=target+dt.timedelta(hours=forecast_steps-1))
    
    # Create and prepare forecasts dataframe
    forecast_df = pd.DataFrame(dict(EURPrices=forecast), columns=["EURPrices"])
    forecast_df.index = pd.date_range(target, periods=forecast.shape[0], freq="H")
    forecast_df.index.name = "DeliveryPeriod"

    return(forecast_df)


"""
Parameters:
    data: pd.DataFrame
    target: datetime
    p: int
        AR model parameter
    trend: str
    forecast_steps: int
"""

def AR_multivariate(data, target, p=None, forecast_steps=24):
    # Not yet sure how to allow for fetching fitted values (in-sample forecasts)
    data = data.copy()
    if "AuctionDateTime" in data.columns:
        data.drop("AuctionDateTime", axis=1, inplace=True)
    
    # Reformat data to pass into VAR
    data_new = pd.DataFrame(columns=pd.unique(data.index.hour),
                           index=pd.unique(data.index.date))
    
    for column in data_new.columns:
        new_column = data.loc[data.index.hour==column]
        new_column.index = new_column.index.date
        data_new[column] = new_column
    
    data_new.index = pd.to_datetime(data_new.index)
    data_new = data_new.asfreq('d').loc[data_new.notna().all(axis=1)]
    
    # Fit VAR(p) model to data
    var_model = tsa.vector_ar.var_model.VAR(endog=data_new, dates=data_new.index, freq='d')
    
    if p is None:
        p = max(var_model.select_order().selected_orders.values())

    var_fit = var_model.fit(maxlags=p)
    
    # Make forecasts
    target = data_new.index[-1] + dt.timedelta(days=1)
    forecast = var_fit.forecast(y=data_new.values, steps=1).flatten()
    
    # Create and prepare forecasts dataframe
    forecast_df = pd.DataFrame(dict(EURPrices=forecast), columns=["EURPrices"])
    forecast_df.index = pd.date_range(target, periods=len(forecast), freq="H")
    forecast_df.index.name = "DeliveryPeriod"

    return(forecast_df)


