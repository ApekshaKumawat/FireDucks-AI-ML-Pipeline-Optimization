import csv
import fireducks as fd
import pandas as pd


data = [
    {'customer_id': 1001, 'age': 25, 'gender': 'Male', 'location': 'NY', 'monthly_spend': 50, 'total_spend': 500, 'num_purchases': 10, 'churn': 'No'},
    {'customer_id': 1002, 'age': 34, 'gender': 'Female', 'location': 'CA', 'monthly_spend': 20, 'total_spend': 200, 'num_purchases': 8, 'churn': 'Yes'},
    {'customer_id': 1003, 'age': 29, 'gender': 'Male', 'location': 'TX', 'monthly_spend': 35, 'total_spend': 400, 'num_purchases': 12, 'churn': 'No'},
    {'customer_id': 1004, 'age': 45, 'gender': 'Female', 'location': 'FL', 'monthly_spend': 75, 'total_spend': 900, 'num_purchases': 15, 'churn': 'No'},
    {'customer_id': 1005, 'age': 38, 'gender': 'Male', 'location': 'IL', 'monthly_spend': 40, 'total_spend': 350, 'num_purchases': 9, 'churn': 'Yes'}
]


# Save data to CSV
with open("data.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)


# Load into a DataFrame
dt = pd.DataFrame(data)
print(dt.head())

data_cleaned = fd.clean_data(dt, strategy='drop')  # Using FireDucks to remove missing values
data_encoded = fd.encode_categorical(data_cleaned, columns=['gender', 'location'], method='onehot')  # Using FireDucks for encoding
data_scaled = fd.scale_features(data_encoded, columns=['age', 'monthly_spend', 'total_spend', 'num_purchases'], method='standard')  # Using FireDucks for scaling

from sklearn.ensemble import RandomForestClassifier
from fireducks.ml import train_model


model = RandomForestClassifier(n_estimators=100)
trained_model = train_model(model, data_scaled, target_column='churn')

