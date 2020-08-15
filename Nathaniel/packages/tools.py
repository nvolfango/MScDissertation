import pandas as pd
import datetime as dt
import time
import re

from functools import reduce

from sklearn import metrics

###########################################################################################################################################
# Availability Proxy data

def read_availabilities_data(filenames, date_format="%Y-%m-%d %H:%M:%S"):
    # Set up date parser
    date_parse1 = lambda date: dt.datetime.strptime(date, date_format) + dt.timedelta(hours=1)
    
    # Read in datasets
    availability_data1 = pd.read_csv(filenames[0], parse_dates=True, index_col="PeriodDateTimes", date_parser=date_parse1)
    
    availability_data2 = pd.read_csv(filenames[1])
    availability_data2.index = pd.to_datetime(availability_data2["PeriodDateTimes"], format="%d/%m/%Y %H:%M") + dt.timedelta(hours=1)
    availability_data2.drop("PeriodDateTimes", axis=1, inplace=True)
                                        
    # Combine datasets
    availability_data = availability_data1.append(availability_data2)
    availability_data.index.name = "DeliveryPeriod"
    
    # Remove duplicates
    availability_data = availability_data.loc[[not val for val in availability_data.index.duplicated()]]
    
    # Remove unnecessary columns
    availability_data.drop(["index"], axis=1, inplace=True)
    
    return(availability_data)

###########################################################################################################################################
# Balancing Market data

def read_bm_price_data(filenames=["Datasets/BMInfo1.csv", "Datasets/BMInfo2.csv", "Datasets/BMInfo3.csv"], columns=["StartTime", "ImbalancePrice"], date_format="%Y-%m-%d %H:%M:%S"):
    # Set up date parser
    date_parse = lambda date: dt.datetime.strptime(date, date_format) + dt.timedelta(hours=1)
    
    # Read in dataset
    bm_data = [pd.read_csv(filenames[i], usecols=columns, parse_dates=True, index_col="StartTime", date_parser=date_parse) for i in range(len(filenames))]
    
    # Combine datasets
    bm_data = reduce(lambda df1, df2: df1.append(df2), bm_data)
    bm_data.freq = "H"
    
    bm_data = bm_data.loc[[not val for val in bm_data.index.duplicated()]]
    
    # Remove last day of data if it does not end at 11 pm
    if bm_data.index.hour[-1] != 23:
        last_date = dt.datetime.combine(bm_data.index.date[-1], dt.datetime.min.time()) - dt.timedelta(hours=1)
        bm_data = bm_data.loc[:last_date]
    
    # Get only hourly data (since BM data has half-hourly granularity)
    bm_data = bm_data.loc[bm_data.index.minute == 0]
    
    # Get list of instances where index is duplicated
    count_df = bm_data.groupby(bm_data.index).count()
    duplicates = count_df.loc[count_df["ImbalancePrice"] != 1].index
    
    # Clean data - remove rows with duplicate indices (by replacing all the rows with their mean)    
    for date in duplicates:
        duplicates_df = bm_data.loc[bm_data.index == date]
        average = duplicates_df.mean().values
        bm_data.drop(date, inplace=True)
        bm_data = bm_data.append(pd.DataFrame(average, index=[date], columns=["ImbalancePrice"]))
    

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
    
    # Rename index
    bm_data.index.name = "DeliveryPeriod"

    return(bm_data)

###########################################################################################################################################
# Demand/Wind forecast data

def read_forecast_data(forecast_type, filename=None, forecast_column=["StartTime", "AggregatedForecast"], date_format="%Y-%m-%d %H:%M:%S"):
    if filename is None:
        if forecast_type == "Wind":
            filename = ["Datasets/WindForecast.csv", "Datasets/WindForecast2.csv"]
        elif forecast_type == "Demand":
            filename = ["Datasets/DemandForecast.csv", "Datasets/DemandForecast2.csv"]
    
    # Set up date parser
    date_parse = lambda date: dt.datetime.strptime(date, date_format) + dt.timedelta(hours=1)
    
    # Read in dataset/s
    forecast_data1 = pd.read_csv(filename[0], usecols=forecast_column, parse_dates=True, index_col="StartTime", date_parser=date_parse)
    forecast_data2 = pd.read_csv(filename[1], usecols=forecast_column, parse_dates=True, index_col="StartTime", date_parser=date_parse)
    
    # Combine datasets
    forecast_data = forecast_data1.append(forecast_data2)
    
    # Remove duplicates
    forecast_data = forecast_data.loc[[not val for val in forecast_data.index.duplicated()]]
    
    # Rename index and column
    forecast_data.index.name = "DeliveryPeriod"
    forecast_data.columns = [forecast_type]
    
    return(forecast_data)

###########################################################################################################################################
# Electricity price data

def read_price_data(filenames=["Datasets/DAMPrices.csv", "Datasets/DAMPrices2.csv"], columns=["DeliveryPeriod", "EURPrices"], date_format="%Y-%m-%d %H:%M:%S"):
    # Set up date parser
    date_parse = lambda date: dt.datetime.strptime(date, date_format) + dt.timedelta(hours=1)
    
    # Read in datasets
    price_data1 = pd.read_csv(filenames[0], usecols=columns, parse_dates=True, index_col="DeliveryPeriod", date_parser=date_parse)
    price_data2 = pd.read_csv(filenames[1], usecols=columns)
    price_data2.index = pd.to_datetime(price_data2["DeliveryPeriod"], format="%d/%m/%Y %H:%M") + dt.timedelta(hours=1)
    price_data2.drop("DeliveryPeriod", axis=1, inplace=True)
    
    # Combine datasets
    price_data = price_data1.append(price_data2)
    price_data.freq = "H"
    
    # Do DST adjustment
    price_data = price_dst_adjustment(price_dst_adjustment(price_data))

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


def price_dst_adjustment(df):
    df_count = df.groupby([df.index.date]).count()
    dst_dates = df_count.loc[(df_count["EURPrices"]) != 24,:]

    if dst_dates.shape[0] == 0:
        return(df)

    for i in range(dst_dates.shape[0]):
        dst_date = dst_dates.index[i]
        number_of_hours = dst_dates.iloc[i,0]

        # Get the price data for the specific dst_date
        df_dst_data = df.loc[df.index.date == dst_date]
        
        # Find the specific hour that's either duplicated (for 25-hour days) or missing (for 23-hour days).
        dst_index = price_find_dst_index(df_dst_data.index.hour, number_of_hours)
        
        # If 23-hour day, get the average of the price data for the adjacent hours.
        if number_of_hours == 23:
            # Fetch adjacent prices, e.g. if missing prices is for 3rd hour, then we fetch prices for 2nd and 4th hour.
            previous_price = df_dst_data.loc[df_dst_data.index.hour == dst_index-1]
            next_day_price = df.loc[df.index.date == dst_date+dt.timedelta(days=1)]
            next_price = next_day_price.loc[next_day_price.index.hour == 0]
            adjacent_prices = previous_price.append(next_price)
            
            # Calculate the average of the two hours of price data
            average_values = adjacent_prices.mean(axis=0).values[0]
            
            # Insert this new price into the original dataframe containing reduced price data.
            new_price = pd.DataFrame(dict(EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            df = df.append(new_price)
            
            # Clean up (reset index)
            df.index = pd.to_datetime(df.index)
            df.index.name = "DeliveryPeriod"
            
        elif number_of_hours == 25:
            # Fetch duplicate prices
            duplicate_prices = df_dst_data.loc[df_dst_data.index.hour == dst_index]

            # Calculate the average of the two hours of price data
            average_values = duplicate_prices.mean(axis=0).values[0]
            
            # Delete the two rows of duplicate hours
            df.drop(duplicate_prices.index, inplace=True)
            
            # Insert this new price into the original dataframe containing reduced price data.
            new_price = pd.DataFrame(dict(EURPrices=average_values), index=[dt.datetime.combine(dst_date, dt.datetime.min.time()) + dt.timedelta(hours=dst_index)])
            df = df.append(new_price)
            df.index = pd.to_datetime(df.index)

    df.sort_index(axis=0, inplace=True)
    df = df.loc[[not val for val in df.index.duplicated()]]
    
    return(df)

###########################################################################################################################################
# Bid Curve data

def read_curve_data(filenames=["Datasets/BidAskCurvve1.csv","Datasets/BidAskCurvve2.csv","Datasets/BidAskCurvve3.csv","Datasets/BidAskCurvve4.csv"], columns=["DeliveryDay", "TimeStepID", "PurchaseVolume", "PurchasePrice", "SellVolume", "SellPrice"]):
    # Read in dataframes into a list
    ba = [pd.read_csv(filenames[i], usecols=columns) for i in range(len(filenames))]
    
    # Combine list of dataframes into a single dataframe
    ba = reduce(lambda df1, df2: df1.append(df2), ba)
    
    # Data cleaning (ensuring consistent data types, remove duplicate days/hours, sort by delivery day/time, etc.)
    ba = clean_ba_data(ba)
    
    return(ba)


""" This function is used in the X-model process when the bid volume data has been divided up according to the price classes
    created during the price class dimension reduction step. Some days were found that were completely missing. The function
    will replace these missing values with the corresponding mean bid volume for that given hour and day of the week (Mon-Sun).
"""

def clean_TimeStepID(TimeStepID):
    # If no need to edit, return the same value
    if type(TimeStepID) == int:
        return(TimeStepID)
    
    # Remove letter/s
    if re.search(".*B$", TimeStepID):
        pos = re.search("B", TimeStepID).start()
    else:
        return(int(TimeStepID))
    
    # Remove letter and convert to int
    return(int(TimeStepID[:pos]))


def clean_ba_data(dataframe):
    # Remove duplicates
    dataframe = dataframe.copy().drop_duplicates()
    
    # Ensure data types are consistent, i.e. DeliveryDay is date, TimeStepID is int, etc.
    dataframe.loc[:,"DeliveryDay"] = pd.to_datetime(dataframe["DeliveryDay"], format="%Y-%m-%d")
    dataframe.loc[:,"TimeStepID"] = dataframe["TimeStepID"].apply(clean_TimeStepID)
    
    dataframe.loc[:,:] = dataframe.sort_values(by=["DeliveryDay","TimeStepID"]).reset_index(drop=True)
    
    # Convert TimeStepID to zero-index, i.e. {0,...,23}
    dataframe.loc[:,"TimeStepID"] = dataframe["TimeStepID"]-1

    # Combine DeliveryDay and TimeStepID into one date column, "DeliveryPeriod", and set as index
    dataframe.loc[:,"DeliveryPeriod"] = [day + dt.timedelta(hours=hour) for (day, hour) in zip(dataframe["DeliveryDay"], dataframe["TimeStepID"])]
    dataframe.set_index("DeliveryPeriod", inplace=True)
    
    # Remove redundant columns
    dataframe.drop(["DeliveryDay", "TimeStepID"], axis=1, inplace=True)
    
    return(dataframe)


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
# Walk-forward evaluation of a forecasting model
"""
Parameters:
    model: model object
    target: DataFrame
        - Given by EURPrices
    bm_data: pd.DataFrame
    planned_data: pd.DataFrame
    starting_window_size: int
        - number of days worth of data to start the training set on
    moving_window: bool
        - To specify whether the training window is a moving window or an expanding window.
    start: datetime
        - Date on which to start the walk-forward validation.
    end: datetime
        - Date to end the walk-forward validation on (inclusive).
"""
def walk_forward_evaluation(model, target, bm_data=None, planned_data=None, starting_window_size=None, moving_window=False, start=None, end=None, logs=True):
    if logs:
        start_time = time.time()
    
    if start is not None and end is not None:
        if start > end:
            raise Exception(f"Cannot have start date after end date.")
    
    # Raise error if training data is not large enough for starting_window_size
    if starting_window_size is not None:
        if (start-target.index[0]).days < starting_window_size:
            raise Exception(f"Not enough data for training: starting_window_size={starting_window_size}, train_size={start-data.index[0]}")
    
    # Create initial training data window
    if starting_window_size is None:
        train_dates = list(pd.date_range(start=target.index[0], end=start-dt.timedelta(hours=1), freq="h"))
    else:
        train_dates = list(pd.date_range(end=start-dt.timedelta(hours=1), periods=24*starting_window_size, freq="h"))

    train_target = target.loc[train_dates,:]
    train_bm_data = None if bm_data is None else bm_data.loc[train_dates,:]
    
    if planned_data is not None:
        if starting_window_size is None:
            train_planned_dates = list(pd.date_range(start=planned_data.index[0], end=start+dt.timedelta(hours=23), freq="h"))
        else:
            train_planned_dates = list(pd.date_range(end=start+dt.timedelta(hours=23), periods=24*(starting_window_size+1), freq="h"))
    
    train_planned_data = None if planned_data is None else planned_data.loc[train_planned_dates,:]
    
    # Create dataframe to store errors
    forecast_index = pd.date_range(start=start, end=end+dt.timedelta(hours=23), freq="h")
    forecasts_df = pd.DataFrame(columns=["Forecast"], index=forecast_index)
    forecasts_df.insert(0, "Original", target["EURPrices"].loc[forecast_index])
    
    # Loop through data to train and forecast iteratively over the expanding (or moving) window
    for _ in range((end-start).days+1):
        model.ingest_data(train_target, train_bm_data, train_planned_data)
        model.train()
        forecast = model.forecast()
        forecasts_df.loc[forecast.index, "Forecast"] = forecast.values
        
        # Drop last day of data if moving_window=True
        if moving_window:
            train_target.drop(train_target.index[:24], inplace=True)
            if bm_data is not None:
                train_bm_data.drop(train_bm_data.index[:24], inplace=True)
            if planned_data is not None:
                train_planned_data.drop(train_planned_data.index[:24], inplace=True)
        
        # Fetch new data and add to the training window
        next_date = list(pd.date_range(start=train_target.index[-1]+dt.timedelta(hours=1), periods=24, freq="h"))
        new_target = target.loc[next_date,:]
        train_target = train_target.append(new_target)
        
        print(f"Finished forecast for {forecast.index.date[0]}.")
        
        if bm_data is not None:
            new_bm_data = bm_data.loc[next_date,:]
            train_bm_data = train_bm_data.append(new_bm_data)
        if planned_data is not None:
            next_planned_date = list(pd.date_range(start=train_planned_data.index[-1]+dt.timedelta(hours=1), periods=24, freq="h"))
            new_planned_data = planned_data.loc[next_planned_date,:]
            train_planned_data = train_planned_data.append(new_planned_data)
            
    rmse = metrics.mean_squared_error(forecasts_df["Original"], forecasts_df["Forecast"], squared=False)
    mae = metrics.mean_absolute_error(forecasts_df["Original"], forecasts_df["Forecast"])
    
    if logs:
        print(f"Execution time: {time.time()-start_time} seconds")
        
    return(model, forecasts_df, rmse, mae)