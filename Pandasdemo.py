import pandas as pd

df = pd.read_csv("data.csv")
print(df) # Display the entire DataFrame
print(df.head()) # Display first 5 rows
print(df.tail()) # Display last 5 rows
print(df.columns) # Display column names
print(df.info()) # Display summary information
print(df.describe()) # Display statistical summary
#print number of rows


