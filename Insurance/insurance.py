import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

datasheet = pd.read_csv('insurance.csv')
print(datasheet)
# print(datasheeet.info())
# print(datasheeet.describe())
# print(datasheet.age.describe())

# age histogram
fig1 = px.histogram(datasheet,
                   title='Distribution of age',
                   x='age',
                   marginal='box',
                   nbins=47)
fig1.update_layout(bargap=0.1)
# fig1.show()

# bmi histogram
fig2 = px.histogram(datasheet,
                    title='Distribution of bmi',
                    marginal='box',
                    color_discrete_sequence=['red'],
                    x='bmi')
fig2.update_layout(bargap=0.1)
# fig2.show()

# charges histogram
fig3 = px.histogram(datasheet,
             marginal='box',
             title='Distribution of charges',
             x='charges',
             color_discrete_sequence=['green', 'gray'],
             color='smoker')
fig3.update_layout(bargap=0.1)
# fig3.show()

fig4 = px.histogram(datasheet,
                    marginal='box',
                    color_discrete_sequence=['blue', 'gray'],
                    color='sex',
                    x='charges',
                    title='Distribution of charges')
fig4.update_layout(bargap=0.1)
# fig4.show()

print(datasheet.smoker.value_counts())

# BMI vs Charges
fig5 = px.scatter(datasheet,
                  title='BMI vs Charges',
                  y='charges',
                  x='bmi',
                  color='smoker',
                  opacity=0.8,
                  hover_data=['sex'])
fig5.update_traces(marker_size=5)
# fig5.show()

# no. of childrens vs charges
fig6 = px.violin(datasheet,
                  title='Charges vs No of childrens',
                  x='children',
                  y='charges')
# fig6.show()

# correlation
print(datasheet.charges.corr(datasheet.age))
print(datasheet.charges.corr(datasheet.bmi))

smoker_values = {'yes': 1, 'no': 0}
smoker_numeric = datasheet.smoker.map(smoker_values) # converting smoker string values to numbers for calculating correlation
# print(smoker_numeric)
print(smoker_numeric.corr(datasheet.charges))

# print(datasheet.corr())

# sns.heatmap(datasheet.corr(), cmap='Reds', annot=True)
# plt.show()

non_smoker_df = datasheet[datasheet.smoker == 'no']
print(non_smoker_df)

plt.title('Age vs Charges')
sns.scatterplot(data=non_smoker_df, x='age', y='charges', alpha=0.7, s=15)
# plt.show()

# plotting a best line in ages vs charges graph
def estimate_charges(age, w, b): # estimate charges calculating using age, w - weight(slope), b - bias(intercept)
    return w*age+b

def try_parameters(w, b): # trying different values for w and b and finding the best one
    ages = non_smoker_df.age
    charge = non_smoker_df.charges

    estimate_charge = estimate_charges(ages, w, b)
    plt.plot(ages, estimate_charge, 'r', alpha=0.9)
    plt.scatter(ages, charge, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual'])
    # calculating Root Mean Square Error(RMSE):
    # RMSE = ROOT(SUM(SQUARE(Predictions - Actual value))/N)
    RMSE = np.sqrt(np.mean(np.square(estimate_charge - charge)))
    print(f'RMSE LOSS: ',RMSE)
    plt.show()

# try_parameters(300, -4000)
# print(non_smoker_df.charges)
# print(estimate_charges(non_smoker_df.age, 300, -4000))
    
# Predicting the models using scikit-learn library (Linear Regression model)
model = LinearRegression()
inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
model.fit(inputs, targets)
model_predictions = model.predict(inputs)
print(model_predictions)
RMSE = np.sqrt(np.mean(np.square(model_predictions - targets)))
print(f'RMSE LOSS: ',RMSE)
print(model.coef_) # w(slope)
print(model.intercept_) # b(intercept)

inputs, targets = non_smoker_df[['age', 'bmi', 'children']], non_smoker_df.charges
model = LinearRegression().fit(inputs, targets)
model_predictions = model.predict(inputs)
print(model_predictions)
RMSE = np.sqrt(np.mean(np.square(model_predictions - targets)))
print(f'RMSE LOSS: ',RMSE)
print(model.coef_) 
print(model.intercept_) 

# barplot for smoker column
sns.barplot(data=datasheet, x='smoker', y='charges')
# plt.show()

# creating a new column which store smoker codes(0 or 1) which represents smoking of a particular person
smoker_codes = { 'yes': 1, 'no': 0 }
datasheet['smoker_code'] = datasheet.smoker.map(smoker_codes)
print(datasheet)
print(datasheet.charges.corr(datasheet.smoker_code))

# creating a new column sex_code which contains binary representation of male or female
sex_codes = { 'male': 1, 'female': 0 }
datasheet['sex_code'] = datasheet.sex.map(sex_codes)
print(datasheet)

# Region plotting
sns.barplot(data=datasheet, x='region', y='charges')
# plt.show()
enc = preprocessing.OneHotEncoder()
enc.fit(datasheet[['region']])
enc.categories_
# print(enc.transform([['northwest']]).toarray())
one_hot = enc.transform(datasheet[['region']]).toarray()
# print(one_hot)
datasheet[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
print(datasheet)

# creting a model for both smoker and non smoker data in the datasheet
inputs, targets = datasheet[['age', 'sex_code', 'bmi', 'children', 'smoker_code', 'northeast', 'northwest', 'southeast', 'southwest']], datasheet.charges
model = LinearRegression().fit(inputs, targets)
model_predictions = model.predict(inputs)
print(model_predictions)
RMSE = np.sqrt(np.mean(np.square(model_predictions - targets)))
print(f'RMSE LOSS: ',RMSE)

print(datasheet[['age', 'sex_code', 'bmi', 'children', 'smoker_code', 'northeast', 'northwest', 'southeast', 'southwest']].loc[10])