import pandas as pd
import datetime as dt

def preprocess_price_data(filename):
    # Read in dataset
    price_data = pd.read_csv(filename)

    # Remove unnecessary columns
    price_data.drop(["AreaSet", "EURGBPRate", "IntervalDuration"], axis=1, inplace=True)

    # Convert dates from string to datetime objects
    price_data["AuctionDateTime"] = pd.to_datetime(price_data["AuctionDateTime"], format="%Y-%m-%d %H:%M:%S")
    price_data["DeliveryPeriod"] = pd.to_datetime(price_data["DeliveryPeriod"], format="%Y-%m-%d %H:%M:%S")

    # Split Delivery period into two columns, one for DeliveryDay, another for Hour ("TimeStepID")
    delivery_day = price_data["DeliveryPeriod"].apply(lambda date: date.strftime("%Y-%m-%d"))
    time_step_id = price_data["DeliveryPeriod"].apply(lambda date: int(date.strftime("%H"))+1)

    # Replace DeliveryPeriod with its component columns (DeliveryDay and TimeStepID)
    price_data.insert(2, "DeliveryDay", delivery_day)
    price_data.insert(3, "TimeStepID", time_step_id)
    price_data["DeliveryDay"] = pd.to_datetime(price_data["DeliveryDay"], format="%Y-%m-%d")
    
    price_data.drop(["DeliveryPeriod"], axis=1, inplace=True)
    
    # Change rows with hour 24 from previous day to hour 0 in the next day (makes the data easier to handle)
    price_data.loc[price_data["TimeStepID"]==24,"DeliveryDay"] = price_data.loc[price_data["TimeStepID"]==24,"DeliveryDay"] + dt.timedelta(days=1)
    price_data["TimeStepID"] = price_data["TimeStepID"].replace({24:0})

    price_data.set_index(["DeliveryDay", "TimeStepID"], inplace=True)
    
    return(price_data)


"""
Parameters:
    data: pd.DataFrame
        - Observed data
    forecast:pd.DataFrame
        - Forecast data
"""
def calculate_errors(data, forecast): 
    # Combine forecast data with the corresponding observed/test/historical data.
    combined_df = forecast.merge(data, how="left", left_index=True, right_index=True, suffixes=("_forecast", "_data"))
    
    # Remove unwanted columns
    unwanted_columns = ["AuctionDateTime"]
    for column in unwanted_columns:
        if column in combined_df.columns:
            combined_df.drop(column, axis=1, inplace=True)
    
    # Calculate residuals, absolute error and squared error of predicted values, respectively.
    combined_df["residual"] = combined_df["EURPrices_data"] - combined_df["EURPrices_forecast"]
    combined_df["absolute_error"] = abs(combined_df["residual"])
    combined_df["squared_error"] = combined_df["residual"] ** 2
    
    return(combined_df)


# Walk-forward validation of a forecasting model
"""
Parameters:
    method_function: function
    parameters: dictionary
        - parameters specific to method_function
    data: pd.DataFrame
        - dataset to train/validate on
    starting_window_size: int
        - number of days worth of data to start the training set on
    moving_window: bool
        - To specify whether the training window is a moving window or an expanding window.
    start: datetime
        - Date on which to start with the training data.
    end: datetime
        - Date to end the walk-forward validation on.
"""
def walk_forward_validation(method_function, parameters, data, starting_window_size, moving_window=False, start=None, end=None):
    # If no dates are passed for start and end, set them as the start date and end date of data
    if start is None:
        start = data.index[0][0]
    elif type(start) == str:
        start = dt.datetime.strptime(start, "%Y-%m-%d")
    elif type(start) == pd._libs.tslibs.timestamps.Timestamp or type(start) == dt.datetime:
        pass
    else:
        raise TypeError("start parameter must be string, datetime or timestamp object.")
    
    if end is None:
        end = data.index[-1][0]
    elif type(end) == str:
        end = dt.datetime.strptime(end, "%Y-%m-%d")
    elif type(end) == pd._libs.tslibs.timestamps.Timestamp or type(end) == dt.datetime:
        pass
    else:
        raise TypeError("end parameter must be string, datetime or timestamp object.")
    
    
    # Create initial training data window
    train_dates = [start + dt.timedelta(days=i) for i in range(starting_window_size)]
    available_data = data.loc[train_dates,:]
    
    # Create dataframe to store errors
    forecast_errors = pd.DataFrame(columns=["DeliveryDay", "TimeStepID", "EURPrices_forecast", "EURPrices_data",
                                            "residual", "absolute_error", "squared_error"])
    forecast_errors.set_index(["DeliveryDay", "TimeStepID"], inplace=True)
    
    
    # Loop through data to train and forecast iteratively over the expanding (or moving) window
    # using the specific model defined by method_function.
    while available_data.index[-1][0] <= end:
        if method_function == naive:
            # Generate forecasts (no training here since naive model doesn't need to be trained)
            forecast_target = train_dates[-1] + dt.timedelta(days=1)
            forecast = naive(data=available_data, target=forecast_target, **parameters)
            
            # Calculate forecast errors
            forecast_error = tools.calculate_errors(data, forecast)
            forecast_errors = forecast_errors.append(forecast_error)
        
        # Extend data window (or move forward, if moving_window=True) by 1 day to include the next day of data
        if moving_window:
            train_dates.pop(0)
        next_date = train_dates[-1] + dt.timedelta(days=1)
        train_dates.append(next_date)
        available_data = data.loc[train_dates,:]
      
        return(forecast_errors)
#     return(forecast_errors)