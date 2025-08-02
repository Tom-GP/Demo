import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

cancer_data = pd.read_csv('breast-cancer.csv')
print(cancer_data)
print(cancer_data.describe())
print(cancer_data.info())
train_val_df, test_df = train_test_split(cancer_data, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
# print(train_df.shape)
# print(val_df.shape)
# print(test_df.shape)
# print(cancer_data.isna().sum())
input_cols = list(train_df)[2:]
output = 'diagnosis'
# print(input_cols)
# print(output)
train_ip = train_df[input_cols].copy()
train_op = train_df[output].copy()
val_ip = val_df[input_cols].copy()
val_op = val_df[output].copy()
test_ip = test_df[input_cols].copy()
test_op = test_df[output].copy()
# print(train_ip)
# print(train_op)
numeric_cols = cancer_data.select_dtypes(include=np.number).columns.tolist()[1:]
# print(numeric_cols)
scaler = MinMaxScaler()
scaler.fit(cancer_data[numeric_cols])
train_ip[numeric_cols] = scaler.transform(train_ip[numeric_cols])
val_ip[numeric_cols] = scaler.transform(val_ip[numeric_cols])
test_ip[numeric_cols] = scaler.transform(test_ip[numeric_cols])
# print(train_ip)
# print(cancer_data['diagnosis'].nunique())
model = LogisticRegression()
model.fit(train_ip, train_op)
# print(model.coef_)
# print(model.intercept_)
model_predict = model.predict(train_ip)
accuracy = accuracy_score(train_op, model_predict)
# print(model_predict)
print(accuracy)
model_predict = model.predict(val_ip)
accuracy = accuracy_score(val_op, model_predict)
# print(model_predict)
print(accuracy)
model_predict = model.predict(test_ip)
accuracy = accuracy_score(test_op, model_predict)
# print(model_predict)
print(accuracy)

data = [
    {
        "id": 857374,
        "radius_mean": 11.94,
        "texture_mean": 18.24,
        "perimeter_mean": 75.71,
        "area_mean": 437.6,
        "smoothness_mean": 0.08261,
        "compactness_mean": 0.04751,
        "concavity_mean": 0.01972,
        "concave points_mean": 0.01349,
        "symmetry_mean": 0.1868,
        "fractal_dimension_mean": 0.0611,
        "radius_se": 0.2273,
        "texture_se": 0.6329,
        "perimeter_se": 1.52,
        "area_se": 17.47,
        "smoothness_se": 0.00721,
        "compactness_se": 0.00838,
        "concavity_se": 0.01311,
        "concave points_se": 0.008,
        "symmetry_se": 0.01996,
        "fractal_dimension_se": 0.002635,
        "radius_worst": 13.1,
        "texture_worst": 21.33,
        "perimeter_worst": 83.67,
        "area_worst": 527.2,
        "smoothness_worst": 0.1144,
        "compactness_worst": 0.08906,
        "concavity_worst": 0.09203,
        "concave points_worst": 0.06296,
        "symmetry_worst": 0.2785,
        "fractal_dimension_worst": 0.07408
    },
    {
        "id": 857392,
        "radius_mean": 18.22,
        "texture_mean": 18.7,
        "perimeter_mean": 120.3,
        "area_mean": 1033,
        "smoothness_mean": 0.1148,
        "compactness_mean": 0.1485,
        "concavity_mean": 0.1772,
        "concave points_mean": 0.106,
        "symmetry_mean": 0.2092,
        "fractal_dimension_mean": 0.0631,
        "radius_se": 0.8337,
        "texture_se": 1.593,
        "perimeter_se": 4.877,
        "area_se": 98.81,
        "smoothness_se": 0.003899,
        "compactness_se": 0.02961,
        "concavity_se": 0.02817,
        "concave points_se": 0.009222,
        "symmetry_se": 0.02674,
        "fractal_dimension_se": 0.005126,
        "radius_worst": 20.6,
        "texture_worst": 24.13,
        "perimeter_worst": 135.1,
        "area_worst": 1321,
        "smoothness_worst": 0.128,
        "compactness_worst": 0.2297,
        "concavity_worst": 0.2623,
        "concave points_worst": 0.1325,
        "symmetry_worst": 0.3021,
        "fractal_dimension_worst": 0.07987
    },
    {
        "id": 857438,
        "radius_mean": 15.1,
        "texture_mean": 22.02,
        "perimeter_mean": 97.26,
        "area_mean": 712.8,
        "smoothness_mean": 0.09056,
        "compactness_mean": 0.07081,
        "concavity_mean": 0.05253,
        "concave points_mean": 0.03334,
        "symmetry_mean": 0.1616,
        "fractal_dimension_mean": 0.05684,
        "radius_se": 0.3105,
        "texture_se": 0.8339,
        "perimeter_se": 2.097,
        "area_se": 29.91,
        "smoothness_se": 0.004675,
        "compactness_se": 0.0103,
        "concavity_se": 0.01603,
        "concave points_se": 0.009222,
        "symmetry_se": 0.01095,
        "fractal_dimension_se": 0.001629,
        "radius_worst": 18.1,
        "texture_worst": 31.69,
        "perimeter_worst": 117.7,
        "area_worst": 1030,
        "smoothness_worst": 0.1389,
        "compactness_worst": 0.2057,
        "concavity_worst": 0.2712,
        "concave points_worst": 0.153,
        "symmetry_worst": 0.2675,
        "fractal_dimension_worst": 0.07873
    },
    {
        "id": 85759902,
        "radius_mean": 11.52,
        "texture_mean": 18.75,
        "perimeter_mean": 73.34,
        "area_mean": 409,
        "smoothness_mean": 0.09524,
        "compactness_mean": 0.05473,
        "concavity_mean": 0.03036,
        "concave points_mean": 0.02278,
        "symmetry_mean": 0.192,
        "fractal_dimension_mean": 0.05907,
        "radius_se": 0.3249,
        "texture_se": 0.9591,
        "perimeter_se": 2.183,
        "area_se": 23.47,
        "smoothness_se": 0.008328,
        "compactness_se": 0.008722,
        "concavity_se": 0.01349,
        "concave points_se": 0.00867,
        "symmetry_se": 0.03218,
        "fractal_dimension_se": 0.002386,
        "radius_worst": 12.84,
        "texture_worst": 22.47,
        "perimeter_worst": 81.81,
        "area_worst": 506.2,
        "smoothness_worst": 0.1249,
        "compactness_worst": 0.0872,
        "concavity_worst": 0.09076,
        "concave points_worst": 0.06316,
        "symmetry_worst": 0.3306,
        "fractal_dimension_worst": 0.07036
    }
]

def BreastCancerDetection(data):
    df = pd.DataFrame(data)
    df = df.drop(columns='id')
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    results = []
    for i in range(len(predictions)):
        pred = predictions[i]
        prob = probabilities[i][list(model.classes_).index(pred)]
        results.append((pred, prob))
    return results
print(BreastCancerDetection(data))