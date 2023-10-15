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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

polygon_api_key = "your polygon.io API key, use 'QUANTGALORE' for 10% off"
calendar = get_calendar("NYSE")

# if you wish to use a different ticker (e.g., CART), it may have been listed after 2006,
# guess and check with the polygon to know the first available date so that you don't waste time with erroneous requests

start_date = "2006-01-01"#(datetime.today() - timedelta(days = 252)).strftime("%Y-%m-%d")
end_date = (datetime.today() - timedelta(days = 1)).strftime("%Y-%m-%d")

trade_dates = pd.DataFrame({"trade_dates": calendar.schedule(start_date = start_date, end_date = end_date).index.strftime("%Y-%m-%d")})

underlying_ticker = "SPY"

volatility_list = []
times = []

for date in trade_dates["trade_dates"]:
    
    try:

        start_time = datetime.now()
        
        underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
        underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time()) & (underlying.index.time < pd.Timestamp("16:00").time())]
        
        if len(underlying) < 350:
            continue
        
        pre_14_session = underlying[underlying.index.hour < 14].copy()
        pre_14_session["returns"] = abs(pre_14_session["c"].pct_change().cumsum())
        
        post_14_session = underlying[underlying.index.hour >= 14].copy()
        post_14_session["returns"] = abs(post_14_session["c"].pct_change().cumsum())
        
        volatility_dataframe = pd.DataFrame([{"date": pd.to_datetime(date), 
                                              "pre_14_vol": round(pre_14_session["returns"].iloc[-1]*100, 2),
                                              "pre_14_volume": round(pre_14_session["v"].sum()),
                                              "post_14_vol": round(post_14_session["returns"].iloc[-1]*100, 2),
                                              "post_14_volume": round(post_14_session["v"].sum())}])
    
        volatility_list.append(volatility_dataframe)
        end_time = datetime.now()
        seconds_to_complete = (end_time - start_time).total_seconds()
        times.append(seconds_to_complete)
        iteration = round((np.where(trade_dates["trade_dates"]==date)[0][0]/len(trade_dates.index))*100,2)
        iterations_remaining = len(trade_dates["trade_dates"]) - np.where(trade_dates["trade_dates"]==date)[0][0]
        average_time_to_complete = np.mean(times)
        estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
        time_remaining = estimated_completion_time - datetime.now()
        
        print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
    except Exception as error:
        print(error)
        continue
    
volatility_dataset = pd.concat(volatility_list).set_index("date")
volatility_dataset["volatility_change"] = volatility_dataset["post_14_vol"] - volatility_dataset["pre_14_vol"]

# how often volatility was higher
len(volatility_dataset[volatility_dataset["volatility_change"] > 0]) / len(volatility_dataset)

training_dataset = volatility_dataset.copy()
training_dataset["year"] = training_dataset.index.year
training_dataset["month"] = training_dataset.index.month
training_dataset["day"] = training_dataset.index.day

features = ["year", "month", "day", "pre_14_volume", "pre_14_vol"]
target = "volatility_change"

# K - Fold Validation

X = training_dataset[features].values
Y = training_dataset[target].apply(Binarizer).values

k_folds = KFold(n_splits = 10, shuffle = False)

RandomForest_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

randomforest_scores = pd.DataFrame(cross_validate(estimator = RandomForest_Model, X=X, y=Y, cv=k_folds, scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]))

#

# Out of sample predictions

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
RandomForest_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X_train, y_train)

random_forest_prediction = RandomForest_Model.predict(X_test)
random_forest_prediction_probability = RandomForest_Model.predict_proba(X_test)

random_forest_prediction_dataframe = pd.DataFrame({"prediction": random_forest_prediction})
random_forest_prediction_dataframe["probability_0"] = random_forest_prediction_probability[:,0]
random_forest_prediction_dataframe["probability_1"] = random_forest_prediction_probability[:,1]
random_forest_prediction_dataframe["probability"] = return_proba(random_forest_prediction_dataframe)
random_forest_prediction_dataframe["actual"] = y_test

tp_rate = len(random_forest_prediction_dataframe[(random_forest_prediction_dataframe["prediction"] == 1) & (random_forest_prediction_dataframe["actual"] == 1)]) / len(random_forest_prediction_dataframe[random_forest_prediction_dataframe["prediction"] == 1])
print(f"True Positive Rate: {tp_rate}")

#

# storing the dataset

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')

# with engine.connect() as conn:
#     result = conn.execute(sqlalchemy.text(f'DROP TABLE vol_dataset'))

training_dataset.to_sql(f"vol_dataset", con = engine, if_exists = "replace")