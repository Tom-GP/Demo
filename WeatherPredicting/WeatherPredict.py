import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

weather_data = pd.read_csv('weatherAUS.csv')
print(weather_data)

# if the data has millon rows, use sample(part of data) for the operations
use_sample = False
sample_fraaction = 0.1
if use_sample:
    weather_data = weather_data.sample(frac = sample_fraaction).copy() # here if use_sample is true then we use only 0.1 of total dataset

# spliting dataframe into training, validation and test datasets
# train_val_df, test_df = train_test_split(weather_data, test_size=0.2, random_state=42)
# train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
# print(train_df.shape)
# print(test_df.shape)
# print(val_df.shape)

# working with time
# plt.title('No of Rows per Year')
# sns.countplot(x=pd.to_datetime(weather_data.Date).dt.year)
# plt.show()

year = pd.to_datetime(weather_data.Date).dt.year # convert to date type and fetch year part and store it to 'year'
train_df = weather_data[year < 2015]
val_df = weather_data[year == 2015]
test_df = weather_data[year > 2015]
# print(train_df.shape)
# print(test_df.shape)
# print(val_df.shape)

# removing the NA values including rows from the RainTomorrow column 
train_df = train_df.dropna(subset=['RainTomorrow'])
val_df = val_df.dropna(subset=['RainTomorrow'])
test_df = test_df.dropna(subset=['RainTomorrow'])

input_cols = list(train_df)[1:-1]
target = 'RainTomorrow'
# print(input_cols)

train_inputs = train_df[input_cols].copy()
train_target = train_df[target].copy()
val_inputs = val_df[input_cols].copy()
val_target = val_df[target].copy()
test_inputs = test_df[input_cols].copy()
test_target = test_df[target].copy()
# print(train_inputs)
# print(train_target)
# print(test_inputs)
# print(test_target)
# print(val_inputs)
# print(val_target)

# print(weather_data.describe())
# print(weather_data.info())

# Getting both numerical and categorical(objects or different types of categories) columns and storing it as list
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
# print(numeric_cols)
# print(categorical_cols)

# print(train_inputs[numeric_cols].describe())
print(train_inputs[categorical_cols].nunique()) # we get no. of different categories of the input sets

# creating a imputer object which will replace the missing NaN values
imputer = SimpleImputer(strategy='mean')

# getting the no. of NaN rows in a given set
# missingvalues = weather_data[numeric_cols].isna().sum()
# print(missingvalues)
print(train_inputs[numeric_cols].isna().sum())
print(val_inputs[numeric_cols].isna().sum())
print(test_inputs[numeric_cols].isna().sum())

imputer.fit(weather_data[numeric_cols]) # calculating all the avg values of each column
print(list(imputer.statistics_)) # printing the avg values

# replacing all the NaN values with the avg values
# print(train_inputs)
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
# print(train_inputs)
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# print(train_inputs[numeric_cols].describe())
# print(weather_data[numeric_cols].describe())

# scaling using MinMaxScaler which is used to identify the max and min value in each column
# scaling all the values in a line ranges 0 to 1
scaler = MinMaxScaler()
scaler.fit(weather_data[numeric_cols]) # Taking the min and max value from each numeric cols
print('Minimum: ', list(scaler.data_min_))
print('Maximum: ', list(scaler.data_max_))
# transforming all the values of each cols in the 3 below sets to the range of 0 to 1
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
# print(train_inputs[numeric_cols].describe())
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# Working with categorical data
print(weather_data[categorical_cols].nunique()) # we get no. of different categories of the input sets
# print(weather_data.Location.unique()) # printing the different categories of Location col

# trasforming categorical colums to numerics which is suitable for ML
# OneHotcoder is used to do this
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(weather_data[categorical_cols])
# raw_df = weather_data[categorical_cols].fillna('Unknown') # replacing the NaN values with Unknown if NaN values is no fit(above code)
print(encoder.categories_) # printing all the different categories in each categorical_cols
encoded_cols = list(encoder.get_feature_names_out(categorical_cols)) # converting each categories to prefix with its categorical_cols name
print(encoded_cols)

# tranforming the dataframe by adding all the different categories created in the above steps as new colums in the 3 sets
encoded_train = pd.DataFrame(
    encoder.transform(train_inputs[categorical_cols]),
    columns=encoded_cols,
    index=train_inputs.index
)

encoded_val = pd.DataFrame(
    encoder.transform(val_inputs[categorical_cols]),
    columns=encoded_cols,
    index=val_inputs.index
)

encoded_test = pd.DataFrame(
    encoder.transform(test_inputs[categorical_cols]),
    columns=encoded_cols,
    index=test_inputs.index
)

# Drop original categorical columns to avoid duplication
train_inputs = train_inputs.drop(columns=categorical_cols)
val_inputs = val_inputs.drop(columns=categorical_cols)
test_inputs = test_inputs.drop(columns=categorical_cols)

# Concatenate encoded features in one shot — avoids fragmentation
train_inputs = pd.concat([train_inputs, encoded_train], axis=1)
val_inputs = pd.concat([val_inputs, encoded_val], axis=1)
test_inputs = pd.concat([test_inputs, encoded_test], axis=1)

# training of logistic model
model = LogisticRegression(solver='liblinear')
model.fit(train_inputs[numeric_cols + encoded_cols], train_target)
# print("Total columns:\n", list(numeric_cols + encoded_cols))
# print("Weight:\n", model.coef_.tolist())

# printing the data colums and the weights in the form of DataFrame
weight_df = pd.DataFrame({
    'feature': (numeric_cols + encoded_cols),
    'weight': model.coef_.tolist()[0]
})
print(weight_df)

# ploting weight and features using barplot
# plt.figure(figsize=(10, 50))
# sns.barplot(data=weight_df.sort_values('weight', ascending=False).head(10), x='weight', y='feature') # important 10 features which will predict RainTomorrow
# plt.show()

print("Intercept:",model.intercept_) # printing the value of intercept

# predicting the values
# creating a variable which contain the predicting 3 set values
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# checking each values predictions
train_predict = model.predict(X_train)
# print(train_predict.tolist())

# checking the accuracy of the predicted value and the target value
print("Accuracy of the model:", accuracy_score(train_target, train_predict))
# chcking the probabilty of RainTomorrow : NO, YES
train_probs = model.predict_proba(X_train)
print("Probabilty: NO  YES\n",  train_probs)

# printing confusion matrix, 4 cases:
# 1) Tomorrow NO rain, predicting NO rain tomarrow - TRUE -ve
# 2) Tomorrow NO rain, predicting YES rain tomarrow - FALSE +ve(Type 1 error)
# 3) Tomorrow YES rain, predicting NO rain tomarrow - FALSE -ve(Type 2 error)
# 1) Tomorrow YES rain, predicting YES rain tomarrow - TRUE +ve
conf_mat = confusion_matrix(train_target, train_predict, normalize='true')
print("Confusion Matrix:\n", conf_mat)

# predicting and plotting the model accuracy and heatmap of confusion matrix with 3 sets
def predict_and_plot(inputs, target, name=''):
    pred = model.predict(inputs)
    accuracy = accuracy_score(target, pred)
    print("Accuracy of the model: {:.2f}%".format(accuracy*100))
    cf = confusion_matrix(target, pred, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    plt.show()
    return pred

# training model
# train_pred = predict_and_plot(X_train, train_target, 'Training')
# print(train_pred)
# validation model
# val_pred = predict_and_plot(X_val, val_target, 'Validation')
# print(val_pred)
# test model
# test_pred = predict_and_plot(X_test, test_target, 'Test')
# print(test_pred)

# validating the model with a given example
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


# eg_df = pd.DataFrame([weather_data_eg])
# print(eg_df)
# eg_df.replace("NA", np.nan, inplace=True)
# eg_df[numeric_cols] = imputer.transform(eg_df[numeric_cols])
# eg_df[numeric_cols] = scaler.transform(eg_df[numeric_cols])
# eg_df[encoded_cols] = encoder.transform(eg_df[categorical_cols])
# print(eg_df)
# X_eg_input = eg_df[numeric_cols + encoded_cols]
# prediction = model.predict(X_eg_input)
# print(prediction)
# prob = model.predict_proba(X_eg_input)
# print(prob)

# creating a function to validate whether rain occur tomorrow or not
def whether_rain_tomorrow(data):
    weather_df = pd.DataFrame([data])
    weather_df.replace("NA", np.nan, inplace=True)
    weather_df[numeric_cols] = imputer.transform(weather_df[numeric_cols])
    weather_df[numeric_cols] = scaler.transform(weather_df[numeric_cols])
    weather_df[encoded_cols] = encoder.transform(weather_df[categorical_cols])
    input = weather_df[numeric_cols + encoded_cols]
    prediction = model.predict(input)[0]
    probability = model.predict_proba(input)[0][list(model.classes_).index(prediction)]
    return prediction, probability

print(whether_rain_tomorrow(weather_data_eg))

# Saving the model using joblib
import joblib
# creating a dic which contains the keys that we want to use the model completely
weather_param = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target': target,
    'numerical_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}
# saving as joblib file
joblib.dump(weather_param, 'weatherPredicting.joblib')

weatherPredict = joblib.load('weatherPredicting.joblib')
print(weatherPredict['model'].predict(X_test))