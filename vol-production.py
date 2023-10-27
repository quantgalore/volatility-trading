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
from sklearn.ensemble import RandomForestClassifier

engine = engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
polygon_api_key = "your polygon.io API key, use 'QUANTGALORE' for 10% off"

vol_dataset = pd.read_sql(sql = "vol_dataset", con = engine).set_index("date")
features = ["year", "month", "day", "pre_14_volume", "pre_14_vol"]
target = "volatility_change"

timeframe = "second"
underlying_ticker = "SPY"
date = datetime.today().strftime("%Y-%m-%d")

training_dataset = vol_dataset[vol_dataset.index < date].copy()

X = training_dataset[features].values
Y = training_dataset[target].apply(Binarizer).values

RandomForest_Model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None).fit(X, Y)

#

underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/{timeframe}/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
underlying = underlying[(underlying.index.time >= pd.Timestamp("09:30").time())].add_prefix("spy_")
underlying["year"] = underlying.index.year
underlying["month"] = underlying.index.month
underlying["day"] = underlying.index.day

pre_14_session = underlying[underlying.index.hour < 14].copy()
pre_14_session["returns"] = abs(pre_14_session["spy_c"].pct_change().cumsum())


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
    print(f"Low volatility expected, {probability*100}% confidence.")
elif prediction == 1:
    print(f"High volatility expected, {probability*100}% confidence.")