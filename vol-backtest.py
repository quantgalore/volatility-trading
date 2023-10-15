# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""

from feature_functions import Binarizer, return_proba

import requests
import pandas as pd
import numpy as np
import sqlalchemy
import mysql.connector
import matplotlib.pyplot as plt

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar
from sklearn.ensemble import RandomForestClassifier

engine = engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
polygon_api_key = "your polygon.io API key, use 'QUANTGALORE' for 10% off"
calendar = get_calendar("NYSE")

vol_dataset = pd.read_sql(sql = "vol_dataset", con = engine).set_index("date")
features = ["year", "month", "day", "pre_14_volume", "pre_14_vol"]
target = "volatility_change"

# the backtest assumes 0-dte options which were launched around this date.
# if you are running this with a different stock, you need to add logic for pulling the available expiration dates
# for help with this, refer to the options implied probability repository where we query all available contracts and expiration dates
start_date = "2022-11-16"
end_date = (datetime.today() - timedelta(days = 1)).strftime("%Y-%m-%d")

trade_dates = pd.DataFrame({"trade_dates": calendar.schedule(start_date = start_date, end_date = end_date).index.strftime("%Y-%m-%d")})

times = []
trades = []

profit_threshold = .10
loss_threshold = -.80

trade_start = pd.Timestamp("14:00").time()
trade_end = pd.Timestamp("16:00").time()

for date in trade_dates["trade_dates"]:

    try:    
        
        start_time = datetime.now()
        
        underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
        underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time()) & (underlying.index.time < trade_end)].add_prefix("spy_")
        underlying["year"] = underlying.index.year
        underlying["month"] = underlying.index.month
        underlying["day"] = underlying.index.day
        
        if len(underlying) < 350:
            continue
        
        underlying["atm_strike"] = round(underlying["spy_c"])
        
        pre_14_session = underlying[underlying.index.hour < 14].copy()
        pre_14_session["returns"] = abs(pre_14_session["spy_c"].pct_change().cumsum())
        
        post_14_session = underlying[underlying.index.hour >= 14].copy()
        post_14_session["returns"] = abs(post_14_session["spy_c"].pct_change().cumsum())
        
        training_dataset = vol_dataset[vol_dataset.index < date].copy()

        X = training_dataset[features].values
        Y = training_dataset[target].apply(Binarizer).values
        
        RandomForest_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X, Y)
        
        production_data = pd.DataFrame([{"year": pre_14_session["year"].iloc[-1], "month": pre_14_session["month"].iloc[-1],
                                          "day": pre_14_session["day"].iloc[-1],
                                          "pre_14_volume": round(pre_14_session["spy_v"].sum()),
                                        "pre_14_vol": round(pre_14_session["returns"].iloc[-1]*100, 2)}])
        
        
        X_prod = production_data[features].values
        
        random_forest_prediction = RandomForest_Model.predict(X_prod)
        random_forest_prediction_probability = RandomForest_Model.predict_proba(X_prod)
        
        random_forest_prediction_dataframe = pd.DataFrame({"prediction": random_forest_prediction})
        random_forest_prediction_dataframe["probability_0"] = random_forest_prediction_probability[:,0]
        random_forest_prediction_dataframe["probability_1"] = random_forest_prediction_probability[:,1]
        random_forest_prediction_dataframe["probability"] = return_proba(random_forest_prediction_dataframe)
        
        prediction = random_forest_prediction_dataframe["prediction"].iloc[0]
        probability = random_forest_prediction_dataframe["probability"].iloc[0]
        
        if prediction == 0:
            continue
        
        trade = post_14_session.head(1).copy()
        
        price = trade["spy_c"].iloc[0]
        returns = pre_14_session["returns"].iloc[-1]
        
        long_put_strike = trade["atm_strike"].iloc[0]
        long_call_strike  = trade["atm_strike"].iloc[0]
        
        # options
        
        Put_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=SPY&contract_type=put&expiration_date={date}&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
        Call_Contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=SPY&contract_type=call&expiration_date={date}&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
        
        Long_Put_Symbol = Put_Contracts[Put_Contracts["strike_price"] == long_put_strike]["ticker"].iloc[0]
        Long_Call_Symbol = Call_Contracts[Call_Contracts["strike_price"] == long_call_strike]["ticker"].iloc[0]
        
        long_put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{Long_Put_Symbol}/range/1/second/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        long_put_ohlcv.index = pd.to_datetime(long_put_ohlcv.index, unit = "ms", utc = True).tz_convert("America/New_York")
    
        long_call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{Long_Call_Symbol}/range/1/second/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        long_call_ohlcv.index = pd.to_datetime(long_call_ohlcv.index, unit = "ms", utc = True).tz_convert("America/New_York")
        
        straddle = pd.concat([long_put_ohlcv.add_prefix("put_"), long_call_ohlcv.add_prefix("call_")], axis  = 1).dropna()
        straddle = straddle[straddle.index.time >= trade_start]
        straddle["straddle_value"] = straddle["put_c"] + straddle["call_c"]
        original_cost = straddle["straddle_value"].iloc[0]
        straddle["straddle_pnl"] = (straddle["straddle_value"] - original_cost) / original_cost
        
        closing_straddle = []
        
        for minute in straddle.index:
            minute_data = straddle[straddle.index == minute].copy()
            current_pnl = minute_data["straddle_pnl"].iloc[0]
            
            if current_pnl >= profit_threshold:
                closing_straddle = minute_data.copy()
                break
            elif current_pnl <= loss_threshold:
                closing_straddle = minute_data.copy()
                break
            # if no trigger event happens by 15:00
            elif (len(closing_straddle) < 1) and minute.time() >= pd.Timestamp("15:00").time():
                closing_straddle = minute_data.copy()
                break
                
        
        open_price = original_cost
        closing_price = closing_straddle["straddle_value"].iloc[0]
        
        gross_pnl = closing_price - open_price
        actual = Binarizer(post_14_session["returns"].iloc[-1] - pre_14_session["returns"].iloc[-1])
        
        trade_dataframe = pd.DataFrame([{"date": pd.to_datetime(date), "prediction": prediction,
                                          "probability": probability, "open_price": open_price,
                                          "closing_price": closing_price, "gross_pnl": gross_pnl,
                                          "actual": actual, "closing_time": closing_straddle.index[0],
                                          "pnl_percent": gross_pnl / original_cost,
                                          "pre_14_vol": pre_14_session["returns"].iloc[-1]*100,
                                          "post_14_vol": post_14_session["returns"].iloc[-1]*100}])
        
        trades.append(trade_dataframe)
        
        end_time = datetime.now()
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(trade_dates["trade_dates"]==date)[0][0]/len(trade_dates.index))*100,2)
        iterations_remaining = len(trade_dates["trade_dates"]) - np.where(trade_dates["trade_dates"]==date)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
    except Exception as error_message:
        print(error_message)
        continue
    

trade_records = pd.concat(trades).set_index("date")
trade_records = trade_records[trade_records["open_price"] <= 2]
trade_records["gross_pnl"] = trade_records["gross_pnl"] * 10
trade_records["net_pnl"] = trade_records["gross_pnl"] -.10
trade_records["capital"] = 1000 + (trade_records["gross_pnl"].cumsum())*100
trade_records["net_capital"] = 1000 + (trade_records["net_pnl"].cumsum())*100

wins = trade_records[(trade_records["gross_pnl"] > 0)].copy()
losses = trade_records[(trade_records["gross_pnl"] < 0)].copy()

average_win = wins["gross_pnl"].mean()
average_loss = losses["gross_pnl"].mean()

monthly_sum = trade_records.resample('M').sum(numeric_only = True)

accuracy_rate = len(trade_records[trade_records["prediction"] == trade_records["actual"]]) / len(trade_records)
win_rate = len(trade_records[trade_records["gross_pnl"] > 0]) / len(trade_records)

expected_value = (win_rate * average_win) + ((1-win_rate) * average_loss)
print(f"\nAccuracy Rate: {round(accuracy_rate*100, 2)}%")
print(f"Win Rate: {round(win_rate*100, 2)}%")
print(f"Expected Value per Trade ${expected_value*100}")
print(f"Average Monthly Profit: ${monthly_sum['gross_pnl'].mean()*100}")

plt.figure(dpi = 200)
plt.xticks(rotation = 45)
plt.title("Buy Straddle When Prediction > 1")

plt.plot(trade_records["capital"])
plt.plot(trade_records["net_capital"])

plt.legend(["gross", "incl. fees"])
plt.show()

##