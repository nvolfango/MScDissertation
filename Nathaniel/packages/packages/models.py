import pandas as pd
import datetime as dt
import numpy as np
import statsmodels.tsa as tsa
import statsmodels.api as sm
import statsmodels.formula.api as smf

"""
Parameters:
    data: pd.DataFrame
    target:
    day_lag: int
        - Number of days to go back in order to make forecast, e.g. day_lag=7 indicates a weekly persistent model,
        day_lag=1 indicates a daily persistent model, etc.
"""
def naive(data, target, day_lag):
    # Create and prepare forecasts dataframe
    forecast_df = pd.DataFrame(columns=["DeliveryDay", "TimeStepID","EURPrices"])
    forecast_df["TimeStepID"] = list(range(24))
    forecast_df["DeliveryDay"] = target
    forecast_df.set_index(["DeliveryDay", "TimeStepID"], inplace=True)

    # Make forecasts (depending on day_lag)
    for i in range(forecast_df.shape[0]):
        delivery_day = forecast_df.index[i][0] - dt.timedelta(days=day_lag)
        delivery_hour = forecast_df.index[i][1]
        try:
            forecast = data.loc[delivery_day, delivery_hour]["EURPrices"].values[0]
        except:
            forecast = data.loc[delivery_day, delivery_hour]["EURPrices"]
        forecast_df["EURPrices"][i] = forecast
    
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
    if "AuctionDateTime" in data.columns:
        data = data.drop("AuctionDateTime", axis=1)
    
    if type(target) == str:
        target = dt.datetime.strptime(target, "%Y-%m-%d")
    
    # Fit AR(p) model to data
    ar_model = tsa.ar_model.AutoReg(data, lags=p, trend=trend).fit()
    
    # Make forecasts
    n = data.shape[0]
    forecast = ar_model.predict(start=n, end=n+forecast_steps-1)
        
    # Create and prepare forecasts dataframe
    forecast_df = pd.DataFrame(dict(EURPrices=forecast), columns=["DeliveryDay", "TimeStepID","EURPrices"])
    forecast_df["TimeStepID"] = list(range(24))
    forecast_df["DeliveryDay"] = target
    forecast_df.set_index(["DeliveryDay", "TimeStepID"], inplace=True)

    return(forecast_df)