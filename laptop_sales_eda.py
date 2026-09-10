import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv("laptop_sales_performance.csv")

# 2. Data understanding
print("\nFIRST 5 ROWS")
print(df.head())

print("\nSHAPE")
print(df.shape)

print("\nCOLUMNS")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nSUMMARY STATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# 3. Date preparation
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])
df["Month"] = df["Sale_Date"].dt.month_name()
df["Month_Number"] = df["Sale_Date"].dt.month

# 4. Univariate Analysis
plt.figure(figsize=(9,5))
sns.countplot(data=df, x="Brand", order=df["Brand"].value_counts().index)
plt.title("Number of Sales by Brand")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
sns.histplot(df["Unit_Price"], bins=30, kde=True)
plt.title("Unit Price Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
sns.histplot(df["Net_Sales"], bins=30, kde=True)
plt.title("Net Sales Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
sns.countplot(data=df, x="Customer_Segment",
              order=df["Customer_Segment"].value_counts().index)
plt.title("Sales by Customer Segment")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 5. Bivariate Analysis
brand_sales = df.groupby("Brand", as_index=False)["Net_Sales"].sum()
brand_sales = brand_sales.sort_values("Net_Sales", ascending=False)

plt.figure(figsize=(9,5))
sns.barplot(data=brand_sales, x="Brand", y="Net_Sales")
plt.title("Total Net Sales by Brand")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

channel_profit = df.groupby("Sales_Channel", as_index=False)["Profit"].sum()

plt.figure(figsize=(9,5))
sns.barplot(data=channel_profit, x="Sales_Channel", y="Profit")
plt.title("Profit by Sales Channel")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
sns.boxplot(data=df, x="RAM_GB", y="Unit_Price")
plt.title("Unit Price by RAM")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
sns.scatterplot(data=df, x="Quantity", y="Net_Sales", hue="Brand")
plt.title("Quantity vs Net Sales")
plt.tight_layout()
plt.show()

# 6. Monthly sales
monthly = df.groupby(["Month_Number","Month"], as_index=False)["Net_Sales"].sum()
monthly = monthly.sort_values("Month_Number")

plt.figure(figsize=(11,5))
sns.lineplot(data=monthly, x="Month", y="Net_Sales", marker="o")
plt.title("Monthly Net Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. City analysis
city_sales = df.groupby("City", as_index=False)["Net_Sales"].sum()
city_sales = city_sales.sort_values("Net_Sales", ascending=False)

plt.figure(figsize=(11,5))
sns.barplot(data=city_sales, x="City", y="Net_Sales")
plt.title("Net Sales by City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Correlation heatmap
numeric_cols = ["RAM_GB","Storage_GB","Screen_Size_Inches","Quantity",
                "Unit_Price","Unit_Cost","Discount_Percent","Gross_Sales",
                "Discount_Amount","Net_Sales","Total_Cost","Profit"]

plt.figure(figsize=(12,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 9. Top 10 models
top_models = df.groupby("Model")["Net_Sales"].sum().sort_values(ascending=False).head(10)
print("\nTOP 10 MODELS BY NET SALES")
print(top_models)

# 10. Final results
print("\nKEY RESULTS")
print(f"Total Units Sold   : {df['Quantity'].sum():,}")
print(f"Total Net Sales    : ₹{df['Net_Sales'].sum():,.2f}")
print(f"Total Profit       : ₹{df['Profit'].sum():,.2f}")
print(f"Average Sale Value : ₹{df['Net_Sales'].mean():,.2f}")

print("\nBEST BRAND")
print(brand_sales.iloc[0])

print("\nBEST CITY")
print(city_sales.iloc[0])

print("\nEDA COMPLETED SUCCESSFULLY")
