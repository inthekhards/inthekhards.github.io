---
title: Group Project
---
Initial Proposal For Team Project
--
Our group decided to analyze 5 different factors on what influences AirBnB profits in NYC. This would allow managers or investors to see what customers were looking for, and which AirBnBs were making the most money and had the most bookings. We compared price, location, room type, neighbourhoods, boroughs, and so on from the labels in the dataset. 

We combined our individual contributions for the final code and paper for submission. We had bi-weekly meetings and more frequent meetings near the end of the project to make sure we were on track. 

Link to code: [Room Type vs. Price](https://github.com/inthekhards/inthekhards.github.io/blob/main/docs/Group_1_Analysis_of_Room_Type_Vs_Price.ipynb) 

Made with Google colab, Unit 2 tutorial code, AB_NYC_2019 dataset, assistance of Gemini, Chat-GPT.

References:

Dgomonov (2019) New York City Airbnb Open Data, www.kaggle.com. Available at: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data.

Google (2024) ‎Gemini - Chat to Supercharge Your Ideas, gemini.google.com. Available at: https://gemini.google.com/.

OpenAI (2025) ChatGPT, ChatGPT. OpenAI. Available at: https://chatgpt.com/.

Tensorflow Authors (2018) Google Colaboratory, colab.research.google.com. Available at: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb.

My Individual Contribution (code section)
--

```
Data Analysis Comparing room_type and price from AB_NYC_2019

# Cleaning up dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("AB_NYC_2019.csv")

print(df['room_type'].unique())

df['room_type'] = df['room_type'].str.strip()
df['room_type'] = df['room_type'].str.lower()

df.isnull().sum()
df = df.dropna(subset=['price', 'room_type'])


room_type_price_stats = df.groupby('room_type')['price'].describe()
room_type_price_stats

# Statistical Analysis of room_type and price

df = pd.read_csv("AB_NYC_2019.csv")

print(df[['room_type', 'price']].isnull().sum())

room_type_price_stats = df.groupby('room_type')['price'].describe()

print(room_type_price_stats)

# 1. Box plot to compare price distribution by room_type

df = df[df['price'] <= 300]
plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='price', data=df)
plt.title('Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()

# 2. Bar plot for average price per room type

avg_price_per_room = df.groupby('room_type')['price'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_per_room.index, y=avg_price_per_room.values)
plt.title('Average Price per Room Type')
plt.xlabel('Room Type')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()

# 3. Histogram of prices by room type

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='price', hue='room_type', kde=True, multiple="stack", bins=50)
plt.title('Price Distribution by Room Type (Histogram)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 4. Pivot table to create a heatmap of price distribution by room type and price bin

price_bins = pd.cut(df['price'], bins=20)

price_bin_room_type = pd.crosstab(price_bins, df['room_type'], normalize='columns')

plt.figure(figsize=(12, 6))
sns.heatmap(price_bin_room_type, cmap='Blues', annot=True, fmt='.2f', linewidths=.5)
plt.title('Heatmap of Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price Range')
plt.show()

# 5. Violin plot for price distribution by room type

plt.figure(figsize=(10, 6))
sns.violinplot(x='room_type', y='price', data=df)
plt.title('Price Distribution by Room Type (Violin Plot)')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()

# 6. KMeans Clustering

X = df[['price']]

kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster'] = kmeans.fit_predict(X)

cluster_summary = df.groupby('cluster')['price'].agg(['mean', 'count'])
print(cluster_summary)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='price', y='room_type', hue='cluster', data=df, palette='viridis')
plt.title('KMeans Clustering of Prices by room_type')
plt.xlabel('Price')
plt.ylabel('room_type')
plt.show()
```
