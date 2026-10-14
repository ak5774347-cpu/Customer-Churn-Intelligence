import pandas as pd

# Load the dataset
df = pd.read_csv("data/customer_data.csv")

# Display the first 5 rows
print("FIRST 5 CUSTOMERS:")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names and data types
print("\nDATASET INFORMATION:")
df.info()

# Check for missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Generate statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nCHURN RATE BY CONTRACT:")

churn_by_contract = df.groupby("Contract")["Churn"].mean() * 100

print(churn_by_contract.round(2))

print("\nCHURN RATE BY PAYMENT METHOD:")

churn_by_payment = df.groupby("PaymentMethod")["Churn"].mean() * 100

print(churn_by_payment.round(2))

print("\nAVERAGE SUPPORT CALLS:")

print(df.groupby("Churn")["SupportCalls"].mean().round(2))


import matplotlib.pyplot as plt

churn_by_contract = df.groupby("Contract")["Churn"].mean() * 100

churn_by_contract.plot(kind="bar")

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.close()

churn_by_payment = df.groupby("PaymentMethod")["Churn"].mean() * 100

churn_by_payment.plot(kind="bar")

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.close()

print("\nAVERAGE REVENUE BY CHURN STATUS:")
print(df.groupby("Churn")["TotalRevenue"].mean().round(2))

print("\nTOTAL REVENUE FROM CHURNED CUSTOMERS:")
print(df[df["Churn"] == 1]["TotalRevenue"].sum().round(2))

print("\nCHURN COUNT:")
print(df["Churn"].value_counts())

print("\nCHURN PERCENTAGE:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))