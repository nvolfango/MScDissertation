import pandas as pd

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

    price_data.insert(2, "DeliveryDay", delivery_day)
    price_data.insert(3, "TimeStepID", time_step_id)
    price_data["DeliveryDay"] = pd.to_datetime(price_data["DeliveryDay"], format="%Y-%m-%d")

    price_data.drop(["DeliveryPeriod"], axis=1, inplace=True)

    return(price_data)