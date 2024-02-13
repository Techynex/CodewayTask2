import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a DataFrame
data = pd.read_csv('D:/task3/fraudTrain.csv', encoding='latin-1')

# Display the column names and shape of the DataFrame
print(data.columns)
print(data.shape)

# Display summary statistics of the DataFrame
print(data.describe())

# Filter the DataFrame to get rows where 'is_fraud' is equal to 1
fraudulent_transactions = data[data['is_fraud'] == 1]

# Display the filtered DataFrame
print(fraudulent_transactions)

# Dataset exploration
print(data.columns)

# Print the shape of the data
data = data.sample(frac=0.1, random_state=1)
print(data.shape)
print(data.describe())

# Plot histograms of each parameter
data.hist(figsize=(20, 20))
plt.show()

# Determine number of fraud cases in dataset
outlier_fraction = len(fraudulent_transactions) / len(data)

print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['is_fraud'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['is_fraud'] == 0])))

# Get all the columns from the dataFrame
columns = data.columns.tolist()

# Filter the columns to remove data we do not want
non_numeric_columns = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
numeric_columns = [c for c in columns if c not in non_numeric_columns and c != 'is_fraud']

# Store the variable we'll be predicting on
target = "is_fraud"

X = data[numeric_columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)

# Correlation matrix
corrmat = X.corr()
fig = plt.figure(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=1),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}

plt.figure(figsize=(9, 7))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Reshape the prediction values to 0 for valid, 1 for fraud.
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

# Ask user for input
print("\nEnter transaction details for prediction:")
user_input = {}
for column in numeric_columns:
    value = input(f"Enter the value for {column}: ")
    user_input[column] = float(value)

# Create a DataFrame from the user input
user_data = pd.DataFrame([user_input])

# Scale the user input
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
user_data_scaled = scaler.transform(user_data)

# Predict using the Isolation Forest model
y_pred = classifiers["Isolation Forest"].predict(user_data_scaled)
if y_pred[0] == 1:
    print("\nPredicted as Fraudulent Transaction.")
else:
    print("\nPredicted as Valid Transaction.")
