import numpy 
import matplotlib.pyplot as plt  
import os 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np
np.random.seed(123)  # For reproducibility


## 
df_name = 'insurance_data_raw.csv'
# Load the insurance data , should be in the same directory as this script
insurance = pd.read_csv(df_name, sep=',')
print(insurance.columns, "\n")
print(insurance.head(), "\n")


# Handle missing values
insurance = insurance.dropna()  # Drop rows with missing values
#insurance = insurance.fillna(insurance.mean())  # Fill missing values with column mean
print("After handling missing values:")
print(insurance.isna().sum())  #get the number of missing values in each column

# Encode string columns to integer
string_columns = insurance.select_dtypes(include='object').columns
label_encoders = {}

for col in string_columns:
    le = LabelEncoder()
    insurance[col] = le.fit_transform(insurance[col])
    label_encoders[col] = le



# Standardize numerical features
insurance["age"] = (insurance["age"] - insurance["age"].mean()) / insurance["age"].std()
insurance["bloodpressure"] = (insurance["bloodpressure"] - insurance["bloodpressure"].mean()) / insurance["bloodpressure"].std()
insurance["bmi"] = (insurance["bmi"] - insurance["bmi"].mean()) / insurance["bmi"].std()

insurance["claim"] = insurance["claim"].astype(int)/max(insurance["claim"])

insurance.to_csv("insurance_data_mod.csv", index=False)
print("Modified data saved to insurance_data_mod.csv")
