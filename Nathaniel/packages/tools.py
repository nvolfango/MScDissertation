import pandas as pd
import datetime as dt

###########################################################################################################################################
# Availability Proxy data

def preprocess_availabilities_data(filename):
    # Read in dataset
    availability_data = pd.read_csv(filename, index_col="PeriodDateTimes")
    
    # Remove unnecessary columns
    availability_data.drop(["index"], axis=1, inplace=True)
    
    # Convert dates from string to datetime objects
    availability_data.index = pd.to_datetime(availability_data.index)
    
    return(availability_data)

###########################################################################################################################################
# Balancing Market data

def preprocess_bm_price_data(filenames):
    # Read in dataset
    bm_data1 = pd.read_csv(filenames[0], usecols=["StartTime", "ImbalancePrice"])
    bm_data2 = pd.read_csv(filenames[1], usecols=["StartTime", "ImbalancePrice"])

    # Combine datasets
    bm_data = bm_data1.append(bm_data2)
    
    # Convert dates from string to datetime objects
    bm_data["StartTime"] = pd.to_datetime(bm_data["StartTime"], format="%Y-%m-%d %H:%M:%S") + dt.timedelta(hours=1)
    
    # Set DeliveryPeriod as index
    bm_data.set_index(["StartTime"], inplace=True)
    bm_data.index.name = "DeliveryPeriod"
    
    # Remove last day of data if it does not end at 11 pm
    if bm_data.index.hour[-1] != 23:
        last_date = dt.datetime.combine(bm_data.index.date[-1], dt.datetime.min.time()) - dt.timedelta(hours=1)
        bm_data = bm_data.loc[:last_date]
    
    # Get only hourly data (since BM data has half-hourly granularity)
    bm_data = bm_data.loc[bm_data.index.minute == 0]
    
    # Get list of instances where index is duplicated
    count_df = bm_data.groupby(bm_data.index).count()
    duplicates = count_df.loc[count_df["ImbalancePrice"] != 1].index
    
    # Clean data - remove rows with dupicate indices (by replacing all the rows with their mean)    
    for date in duplicates:
        duplicates_df = bm_data.loc[bm_data.index == date]
        average = duplicates_df.mean().values
        bm_data.drop(date, inplace=True)
        bm_data = bm_data.append(pd.DataFrame(average, index=[date], columns=["ImbalancePrice"]))
    bm_data.index.name = "DeliveryPeriod"
    
    # Get list of dates that do not have exactly 24 data points (hours)
    count_df = bm_data.groupby(bm_data.index.date).count()
    missing_values = count_df.loc[count_df["ImbalancePrice"] != 24]
    
    # Clean data - Insert missing values with the overall mean
    for date in missing_values.index:
        date_hours = bm_data.loc[bm_data.index.date == date].index.hour

        for hour in range(24):
            if hour not in date_hours:
                missing_value_loc = dt.datetime.combine(date, dt.datetime.min.time()) + dt.timedelta(hours=hour)
                imputed_value = pd.DataFrame(columns=["ImbalancePrice"], index=[missing_value_loc])
                bm_data = bm_data.append(imputed_value)
            
    bm_data.fillna(bm_data.mean(), inplace=True)
    bm_data.sort_index(inplace=True)
    
    return(bm_data)

###########################################################################################################################################
# Demand/Wind forecast data

def preprocess_forecast_data(filename, forecast_column="AggregatedForecast"):
    # Read in dataset
    forecast_data = pd.read_csv(filename)

    # Remove unnecessary columns
    forecast_data.drop(["PublishTime", "EndTime"], axis=1, inplace=True)

    # Convert dates from string to datetime objects
    forecast_data["StartTime"] = pd.to_datetime(forecast_data["StartTime"], format="%Y-%m-%d %H:%M:%S") + dt.timedelta(hours=1)
    
    # Set DeliveryPeriod as index
    forecast_data.set_index(["StartTime"], inplace=True)
    forecast_data.index.name = "DeliveryPeriod"
    
    # Get forecast_column
    if type(forecast_column) == str:
        forecast_data = forecast_data[[forecast_column]]
    else:
        forecast_data = forecast_data[forecast_column]
    
    return(forecast_data)

###########################################################################################################################################
# Electricity price data

def preprocess_price_data(filename):
    # Read in dataset
    price_data = pd.read_csv(filename)

    # Remove unnecessary columns
    price_data.drop(["AreaSet", "EURGBPRate", "IntervalDuration"], axis=1, inplace=True)

    # Convert dates from string to datetime objects
    price_data["AuctionDateTime"] = pd.to_datetime(price_data["AuctionDateTime"], format="%Y-%m-%d %H:%M:%S")
    price_data["DeliveryPeriod"] = pd.to_datetime(price_data["DeliveryPeriod"], format="%Y-%m-%d %H:%M:%S") + dt.timedelta(hours=1)
    
    # Set DeliveryPeriod as index
    price_data.set_index(["DeliveryPeriod"], inplace=True)
    
    return(price_data)


def price_find_dst_index(time_step_id_dataframe, number_of_hours):
    if number_of_hours == 23:
        for i, time_step_id in enumerate(time_step_id_dataframe):
            if i < time_step_id:
                return(i+1)
            elif i == number_of_hours-1:
                return(23)
                
    elif number_of_hours == 25:
        for j, time_step_id in enumerate(time_step_id_dataframe):
            if j+1 > time_step_id:
                return(time_step_id)


def price_dst_adjustment(df, adt_column=True):
    df_count = df.groupby([df.index.date]).count()
    dst_dates = df_count.loc[(df_count["EURPrices"]) != 24,:]
    
    if dst_dates.shape[0] == 0:
        return(df)
    
    if adt_column:
        dst_dates.drop("AuctionDateTime", axis=1, inplace=True)
    
    for i in range(dst_dates.shape[0]):
        dst_date = dst_dates.index[i]
        number_of_hours = dst_dates.iloc[i,0]

        # Get the price data for the specific dst_date
        df_dst_data = df.loc[df.index.date == dst_date]
        if adt_column:
            adt = df_dst_data["AuctionDateTime"].values[-1]
        
        # Find the specific hour that's either duplicated (for 25-hour days) or missing (for 23-hour days).
        dst_index = price_find_dst_index(df_dst_data.index.hour, number_of_hours)

        # If 23-hour day, get the average of the price data for the adjacent hours.
        if number_of_hours == 23:
            # Fetch adjacent prices, e.g. if missing prices is for 3rd hour, then we fetch prices for 2nd and 4th hour.
            previous_price = df_dst_data.loc[df_dst_data.index.hour == dst_index-1]
            next_day_price = df.loc[df.index.date == dst_date+dt.timedelta(days=1)]
            next_price = next_day_price.loc[next_day_price.index.hour == 0]
            if adt_column:
                adjacent_prices = previous_price.append(next_price).drop("AuctionDateTime", axis=1)
            else:
                adjacent_prices = previous_price.append(next_price)
            
            # Calculate the average of the two hours of price data
            average_values = adjacent_prices.mean(axis=0).values[0]
            
            # Insert this new price into the original dataframe containing reduced price data.
            if adt_column:
                new_price = pd.DataFrame(dict(AuctionDateTime=adt, EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            else:
                new_price = pd.DataFrame(dict(EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            df = df.append(new_price)
            
            # Clean up (reset index)
            df.index = pd.to_datetime(df.index)
            df.index.name = "DeliveryPeriod"
            
            
        elif number_of_hours == 25:
            # Fetch duplicate prices
            duplicate_prices = df_dst_data.loc[df_dst_data.index.hour == dst_index]
            if adt_column:
                duplicate_prices.drop(["AuctionDateTime"], axis=1, inplace=True)

            # Calculate the average of the two hours of price data
            average_values = duplicate_prices.mean(axis=0).values[0]
            
            # Delete the two rows of duplicate hours
            df.drop(duplicate_prices.index, inplace=True)
            
            # Insert this new price into the original dataframe containing reduced price data.
            if adt_column:
                new_price = pd.DataFrame(dict(AuctionDateTime=adt, EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            else:
                new_price = pd.DataFrame(dict(EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            df = df.append(new_price)
            df.index = pd.to_datetime(df.index)

    df.sort_index(axis=0, inplace=True)
    
    return(df)

###########################################################################################################################################
# Bid Curve data

""" This function is used in the X-model process when the bid volume data has been divided up according to the price classes
    created during the price class dimension reduction step. Some days were found that were completely missing. The function
    will replace these missing values with the corresponding mean bid volume for that given hour and day of the week (Mon-Sun).
"""

def clean_reduced_bid_data(dataframe):
    df = dataframe.copy()
    
    # Insert missing dates (as NaNs) into the dataframe
    start_date = df.index[0]
    end_date = df.index[-1]
    df = df.reindex(pd.date_range(start_date, end_date, freq="H"))
    
    # Calculate overall mean for each given hour of each given day (Monday to Sunday)
    df["DayOfWeek"] = df.index.dayofweek
    means = df.groupby(["DayOfWeek", df.index.hour]).mean()
    
    # Fetch list of missing dates
    missing_dates = df.loc[df.isna().any(axis=1)].index

    # Replace missing values with appropriate mean value
    for date in missing_dates:
        day_of_week = df["DayOfWeek"].loc[date]
        hour = date.hour
        mean = means.loc[day_of_week, hour]
        df.loc[date] = mean
    
    return(df.drop("DayOfWeek", axis=1))

###########################################################################################################################################
# RandomForest predictor arrangement

# def organise_predictors():



###########################################################################################################################################
# Model forecast results

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