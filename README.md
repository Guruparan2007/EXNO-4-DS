# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
~~~
Name: GURUPARAN G

Register no:212224220030
~~~

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![Screenshot 2025-04-18 163404](https://github.com/user-attachments/assets/9cd6916b-e75d-4741-9917-f488f977ec1f)
```
df.dropna()
```
![Screenshot 2025-04-18 163441](https://github.com/user-attachments/assets/d7b28fdd-8c9c-4800-b775-14c9d90c3011)
```
max_vals = df[['Height', 'Weight']].abs().max()
print(max_vals)
```
![Screenshot 2025-04-18 163452](https://github.com/user-attachments/assets/8e16182d-3e58-4d77-9388-1d89f5e84066)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2025-04-18 163458](https://github.com/user-attachments/assets/e60ae8bb-e5d5-46ad-9a2c-01868cfd1b0d)
```
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
df[['Height','Weight']]=scalar.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2025-04-18 163503](https://github.com/user-attachments/assets/ee3e1ba3-d3ae-40f3-95c5-c79ebd2f2002)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2025-04-18 163510](https://github.com/user-attachments/assets/38e8be0c-4d24-4488-8dc3-998cd32ce6a7)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2025-04-18 163517](https://github.com/user-attachments/assets/1aabc487-df88-4d97-9783-8a88292f6d0d)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2025-04-18 163525](https://github.com/user-attachments/assets/66a888df-8759-4e41-a02a-40a33f812458)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2025-04-18 163532](https://github.com/user-attachments/assets/15bfbabe-3822-4009-86ce-c42bc6acf6cb)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2025-04-18 163538](https://github.com/user-attachments/assets/6386462d-eaf9-487f-a2b6-653292a89343)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-squared statistic: {chi2}")
print(f"P-value: {p}")
```
![Screenshot 2025-04-18 163542](https://github.com/user-attachments/assets/41c5d831-fced-4809-9d8e-719156fde396)
```
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2025-04-18 163550](https://github.com/user-attachments/assets/2b15edc7-590e-4d02-b516-b373c090cb6b)

# RESULT:

Successfully read the given data and performed Feature Scaling and Feature Selection process and saveed the
data to a file.
