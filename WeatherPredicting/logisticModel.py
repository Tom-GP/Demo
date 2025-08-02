import joblib
import pandas as pd

# loading joblib ML weatherPredicting file
weatherPredict = joblib.load('weatherPredicting.joblib')

weather_data_eg = {
    "Date": "2008-12-02",
    "Location": "Albury",
    "MinTemp": 14.0,
    "MaxTemp": 28.0,
    "Rainfall": 3.6,
    "Evaporation": "NA",
    "Sunshine": "NA",
    "WindGustDir": "WNW",
    "WindGustSpeed": 44,
    "WindDir9am": "NNW",
    "WindDir3pm": "WSW",
    "WindSpeed9am": 24,
    "WindSpeed3pm": 22,
    "Humidity9am": 44,
    "Humidity3pm": 25,
    "Pressure9am": 1010.6,
    "Pressure3pm": 1007.5,
    "Cloud9am": 4,
    "Cloud3pm": 2,
    "Temp9am": 17.2,
    "Temp3pm": 26.5,
    "RainToday": "No"
}

weather_data_eg = pd.DataFrame([weather_data_eg])
print(weatherPredict['model'].predict(weather_data_eg))