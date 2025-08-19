# Building a machine learning to identify churned users | Python
Author: Nguyễn Hải Long  
Date: 2025-05  
Tools Used: SQL 

---

## 📑 Table of Contents  
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)  
3. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

---

## 📌 Background & Overview  

### Objective:
### 📖 This project is about using SQL to analyze transaction data from Google analytic dataset.  

- The dataset captures user behavior and business performance metrics of an e-commerce website in 2017, including traffic volume, conversion rates, traffic sources, revenue, and cross-sell products.  
- The main objective is to analyze the customer journey and evaluate the effectiveness of marketing channels to optimize user experience and drive revenue growth.  

### 👤 Who is this project for?  

- Data analysts & business analysts.  
- Decision-makers.

---

## 📂 Dataset Description & Data Structure  

### 📌 Data Source  
- Source: Company database.  
- Size: The dataset has 20 columns and 5630 rows.
- Format: .xlsx  

### 📊 Data Structure & Relationships  
#### 1️⃣ Table used: 
Using the whole dataset.  

#### 2️⃣ Table Schema & Data Snapshot:  
<details>
 <summary>Table using in this project:</summary>

| Field Name | Data Type |
|------------|-----------|
| CustomerID | int64 |
| Churn | int64 |
| Tenure | float64 |
| PreferredLoginDevice | object |
| CityTier | int64 |
| WarehouseToHome | float64 |
| PreferredPaymentMode | object |
| Gender | object |
| HourSpendOnApp | float64 |
| NumberOfDeviceRegistered | int64 |
| PreferredOrderCat | object |
| SatisfactionScore | int64 |
| MaritalStatus | object |
| NumberOfAddress | int64 |
| Complain | int64 |
| OrderAmountHikeFromlastYear | float64 |
| CouponUsed | float64 |
| OrderCount | float64 |
| DaySinceLastOrder | float64 |
| CashbackAmount | float64 |

</details>

- Sheet 'ecommerce retail' will provide data for EDA, calculating.  
- Using sheet 'Segmentation' for segment customers based on the score.

---

## ⚒️ Main Process  

*Note: Click the white triangle to see codes*  

### 1️⃣ EDA
<details>
 <summary><strong>Import libraries and dataset, copy dataset:</strong></summary>
  
  ```python

  !pip install category_encoders

  # import libraries
  import pandas as pd
  import numpy as np
  from google.colab import drive
  import math
  import category_encoders as ce
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import balanced_accuracy_score
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import mean_squared_error, r2_score
  from sklearn.decomposition import PCA
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # import excel files
  drive.mount('/content/drive')
  path = '/content/drive/MyDrive/DAC K34/Machine Learning/Final project/churn_prediction.xlsx'
  
  df = pd.read_excel(path)
  
  ```
</details>  

#### Understanding data    

<details>
 <summary><strong>Basic data exploration:</strong></summary>

 ```python
 df.head()
 
 # show rows and columns count
 print(f'Rows count: {df.shape[0]}\nColums count: {df.shape[1]}')
```

![]()

```python
 # show data type
 df.info()
```

![]()

```python
 # further checking on columns
 df.describe()
```

|  | CustomerID | Churn | Tenure | CityTier | WarehouseToHome | HourSpendOnApp | NumberOfDeviceRegistered | SatisfactionScore | NumberOfAddress | Complain | OrderAmountHikeFromlastyear | CouponUsed | OrderCount | DaySinceLastOrder | CashbackAmount |
| count | 5630 | 5630 | 5366 | 5630 | 5379 | 5375 | 5630 | 5630 | 5630 | 5630 | 5365 | 5374 | 5372 | 5323 | 5630 |
| mean | 52815.5 | 0.168384 | 10.189899 | 1.654707 | 15.639896 | 2.931535 | 3.688988 | 3.066785 | 4.214032 | 0.284902 | 15.707922 | 1.751023 | 3.008004 | 4.543491 | 177.223030 |
| std | 1625.385339 | 0.374240 | 8.557241 | 0.915389 | 8.531475 | 0.721926 | 1.023999 | 1.380194 | 2.583586 | 0.451408 | 3.675485 | 1.894621 | 2.939680 | 3.654433 | 49.207036 |
| min | 5001 | 0 | 0 | 1 | 5 | 0 | 1 | 1 | 1 | 0 | 11 | 0 | 1 | 0 | 0 |
| 25% | 51408.25 | 0 | 2 | 1 | 9 | 2 | 3 | 2 | 2 | 0 | 13 | 1 | 1 | 2 | 145.77 |
| 50% | 52815.5 | 0 |
| 75% | 54222.75 | 0 |
| max | 55630 | 0 |

```python
 # check null values
 df.isnull().sum()
 
 # checking unique values
 ## percentage of unique values
 num_unique = df.nunique().sort_values()
 print('---Percentage of unique values (%)---')
 print(100/num_unique)
 
 # check missing data
 missing_value = df.isnull().sum().sort_values(ascending = False)
 missing_percent = df.isnull().mean().sort_values(ascending = False)
 print('')
 print('---Number of missing values in each column---')
 print(missing_value)
 print('')
 print('---Percentage of missing values (%)---')
 if missing_percent.sum():
   print(missing_percent[missing_percent > 0] * 100)
 else:
   print('None')
 
 # check for duplicates
 ## show number of duplicated rows
 print('')
 print(f'Number of entirely duplicated rows: {df.duplicated().sum()}')
 ## show all duplicated rows
 df[df.duplicated()]
 ```

![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_1.png)  
![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_2.png)

</details>

➡️ This dataset has 8 columns and 541,909 records. Most columns have right data type:  
- InvoiceNo (object) -> change to string data type for further handling.  
- CustomerID (float64) -> can be changed to int64 if needed.  
  
➡️ The percentage of duplicated values is acceptable. Missing values in "CustomerID" are high (~25%), it will affect the analysis. They need to verify and fill up as much as possible. 5268 rows of duplicating contain duplicated information of "Quantity", "InvoiceDate", "CustomerID", "Country". These rows are acceptable because there will be a customer buying many products in a day from any country.  

<details>
 <summary><strong>Change data type of 'InvoiceNo' to string:</strong></summary>

 ```python
 # change data type of Invoice No to string
 df['InvoiceNo'] = df['InvoiceNo'].astype(str)
 ```

</details>

➡️ The purpose for this action: Easy for handling duplicated values.

<details>
 <summary><strong>Explore negative values of Quantity columns (Quantity < 0 and UnitPrice < 0):</strong></summary>
  
 ```python
 # print out some rows where Quantity < 0
 print('Some rows have Quantity < 0')
 print(df[df['Quantity']<0].head())
 
 
 # further checking
 ## make a new column: True if InvoiceNo has 'C', False if InvoiceNo has no 'C'
 df['Cancellation'] = df['InvoiceNo'].str.contains('C')
 
 ## check InvoiceNo has 'C' and Quantity < 0
 print(df[(df['Cancellation'] == True) & (df['Quantity'] < 0)].head())
 print('asoidfbao',df['CustomerID'].isna().sum())
 
 ## check InvoiceNo has no 'C' and Quantity < 0
 print(df[(df['Cancellation'] == False) & (df['Quantity'] < 0)].head())
 ```

 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_3.png)

 ```python
 # print out some rows where Quantity < 0
 print('Some rows have UnitPrice < 0')
 print(df[df['UnitPrice'] < 0].head())
 ```

![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_4.png)
 
</details>

➡️ There are two reasons behind Quantity < 0:
- Orders with InvoiceNo has C are cancelled orders.
- Rows with UnitPrice = 0 are returned orders.

➡️ Orders with UnitPrice < 0 are in "Adjust bad dept" state as noted in "Description" column.  
➡️ We can drop these rows to segment customers precisely.  

<details>
 <summary><strong>Seperate "InvoiceDate" to "Day" and "Month" columns:</strong></summary>
  
 ```python
 # seperate InvoiceDate to Day and Month columns
 df['Day'] = pd.to_datetime(df.InvoiceDate).dt.date
 df['Month'] = df['Day'].apply(lambda x: str(x)[:-3])
 df.head()
 ```

 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_5.png)

</details>

➡️ The 'InvoiceDate' column is split into 'Day' and 'Month' to later identify the customer's most recent interaction date, which is essential for calculating the Recency metric.  

#### Handle negative, missing values, duplicates:  

<details>
 <summary><strong>Negative values:</strong></summary>
  
 ```python
 # change data type
 df['StockCode'] = df['StockCode'].astype(str)
 df['Description'] = df['Description'].astype(str)
 df['CustomerID'] = df['CustomerID'].astype(str)
 df['Country'] = df['Country'].astype(str)
 
 # drop negative values in Quantity and UnitPrice column
 df = df[df['Quantity'] > 0]
 df = df[df['UnitPrice'] > 0]
 
 # drop InvoiceNo with C
 df = df[df['Cancellation'] == False]
 
 # replace NaN
 df = df.replace('nan', None)
 df = df.replace('Nan', None)
 
 df.info()
 ```

 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_6.png)

</details>

➡️ Remove all negative values in 'Quantity', 'UnitPrice' and 'InvoiceNo' with 'C' because they are cancelled orders.  

<details>
 <summary><strong>Missing values:</strong></summary>
  
 ```python
 # show up some rows with missing values
 print('---Some rows with missing values---')
 df_null = df.isnull()
 rows_with_null = df_null.any(axis=1)
 df_with_null = df[rows_with_null]
 print(df_with_null.head(10))
 ```
 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_7.png)
 
 ```python
 # drop rows with CustomerID == None
 df_no_na = df.drop(df[df['CustomerID'].isnull()].index)
 df_no_na
 ```

 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_8.png)

</details>

➡️ Drop all rows with 'CustomerID' is null. The reason for this action is cannot identify the customers.  

<details>
 <summary><strong>Duplicated values:</strong></summary>
  
 ```python
 # locate the values are not duplicated in the selected columns
 df_no_dup = df_no_na.loc[~df.duplicated(subset = ['InvoiceNo','StockCode','InvoiceDate','UnitPrice','CustomerID','Country'])].reset_index(drop=True).copy()
 
 # check an example of duplicate in InvoiceNo
 df_no_dup.query('InvoiceNo == "536365"')
 
 df_no_dup.query('InvoiceNo == "581587"')
 ```
 
 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_9.png)
 
 ```python
 # drop duplicates, keep the first row of subset
 df_main = df.drop_duplicates(subset=["InvoiceNo", "StockCode","InvoiceDate","CustomerID"], keep = 'first')
 
 df_main.head()
 ```

 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_10.png)

</details>

➡️ In this step, we drop all duplicated rows with same information from all columns "InvoiceNo", "StockCode", "InvoiceDate", "UnitPrice", "CustomerID", "Country". Then with the remaining result, keeping only the first rows for R-F-M calculation.  

<details>
 <summary><strong>Create 'Sales' column (Quantity * Price):</strong></summary>
  
 ```python
 # create Sales column (Quantity * UnitPrice)
 df_main['Sales'] = df_main.Quantity * df.UnitPrice
 
 # take max('Day') for recently interaction of customer
 last_day = df_main.Day.max()
 
 last_day
 df_main
 ```

 ![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_eda_11.png)

</details>

➡️ Taking max of 'Day' in order to identify the most recent date of interaction of customers.  

### 2️⃣ Data processing   

<details>
 <summary><strong>Handle Segmentation table</strong></summary>  

 ```python
 # import excel files with sheet name 'Segmentation'
 segmentation = pd.read_excel (path, sheet_name ='Segmentation')
 
 # copy dataframe
 df_seg = segmentation
 
 # transform Segmentation
 df_seg['RFM Score'] = df_seg['RFM Score'].str.split(',')
 df_seg = df_seg.explode('RFM Score').reset_index(drop=True)
 
 df_seg.head()
 ```

 ![data_processing_1](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_data_processing_1.png)  

</details>

➡️ The Segmentation copy process involves duplicating a new Segmentation table to avoid interference with the original dataset, thereby preventing unintended data modifications. The transformation of the Segmentation table will split segments based on predefined RFM scores. These scores are currently separated by commas, so this process will parse them into the required segments accordingly.  
<details>
 <summary><strong>Calculating RFM</strong></summary>
 
 ```python
 # determining Recency, Frequency, Monetary
 df_RFM = df_main.groupby('CustomerID').agg(
     Recency = ('Day', lambda x: last_day - x.max()),
     Frequency = ('CustomerID','count'),
     Monetary = ('Sales','sum'),
     Start_Date = ('Day','min')
 ).reset_index()
 
 df_RFM['Recency'] = df_RFM['Recency'].dt.days.astype('int16')
 # take opposite of Recency
 df_RFM['Reverse_Recency'] = -df_RFM['Recency']
 df_RFM['Start_Date'] = pd.to_datetime(df_RFM.Start_Date)
 
 # label R, F, M
 df_RFM['R'] = pd.qcut(df_RFM['Reverse_Recency'], 5, labels = range(1,6)).astype(str)
 df_RFM['F'] = pd.qcut(df_RFM['Frequency'], 5, labels = range(1,6)).astype(str)
 df_RFM['M'] = pd.qcut(df_RFM['Monetary'], 5, labels = range(1,6)).astype(str)
 df_RFM['RFM'] = df_RFM.R + df_RFM.F + df_RFM.M
 
 df_RFM.head()
 ```
 ![data_processing_2](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_data_processing_2.png)

 ```python
 # clear space
 df_seg['RFM Score'] = df_seg['RFM Score'].str.strip()
 
 # merge with Segementation for comparison
 df_RFM_final = df_RFM.merge(df_seg, how='left', left_on='RFM', right_on='RFM Score')
 
 df_RFM_final.head()
 ```

 ![data_processing_3](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_data_processing_3.png)

</details>

**In this stage, RFM is calculated:**  
  1. Recency is computed as the last purchase date minus the dataset’s maximum date, the low value the better. However, the convenience in label, we use negative value of Recency. That means **the bigger the better**, and ranking is from 1 = worst to 5 = best.  
  2. Frequency measures how often a customer makes a purchase and is computed as counting the number of appearance of each customer, **the bigger the better**.  
  3. Monetary represents the total of money spending from each customer, **the bigger the better**.  
Afterward, the results of the three metrics are assigned scores on a scale from 1 to 5.
In the final step, the combined RFM scores are matched against the Segmentation table to assign each customer to a corresponding segment.  

<details>
 <summary><strong>Determine Loyal and Non Loyal and showing characteristic of Potential Loyalist:</strong></summary>

 ```python
 df_RFM_final['Loyal_Status'] = df_RFM_final['Segment'].apply(lambda x: 'Loyal' if x in ('Loyal','Potential Loyalist') else 'Non Loyal')

 df_RFM_final.head()
 ```

 ![data_processing_4](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_data_processing_4.png)

➡️ Determining "Loyal" and "Non Loyal" state based on Segmentation table.

</details>

<details>
 <summary><strong>Creating df_RFM_final for visualization:</strong></summary>
 
 ```python
 # Average of Quantity and Sales according to CustomerID
 df_potential_average = df_main.groupby('CustomerID').agg(
     Quantity_Average = ('Quantity','mean'),
     Sales_Average = ('Sales','mean')
 ).reset_index()
 
 df_potential_average.head()
 ```

 ![data_processing_5](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_data_processing_5.png)

 ```python
 # First Sales and Quantity according to CustomerID
 ## base on InvoiceDate to get first order -> Quantity, Sales
 df_main['Ranking'] = df_main.groupby('CustomerID')['InvoiceDate'].rank(method = 'first')
 df_potential_first = df_main[df_main.Ranking == 1][['CustomerID','Quantity','Sales']]
 df_potential_first = df_potential_first.rename(columns={'Quantity':'First_Quantity','Sales':'First_Sales'})
 
 df_potential_first.head()
 ```
 
 ![data_processing_6](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_data_processing_6.png)

 ```python
 # merge all
 df_RFM_final = df_RFM_final.merge(df_potential_average, how = 'left', on = 'CustomerID')
 df_RFM_final = df_RFM_final.merge(df_potential_first, how = 'left', on = 'CustomerID')
 
 df_RFM_final.head()
 ```
</details>

|  | CustomerID | Recency | Frequency | Monetary | Start_Date | Reverse_Recency | R | F | M | RFM | Segment | RFM Score | Loyal_Status | Quantity_Average | Sales_Average | First_Quantity | First_Sales |
|---|-----------|---------|-----------|----------|------------|-----------------|---|---|---|-----|---------|-----------|--------------|------------------|---------------|----------------|--------|
| 0 | 12346.0 | 325 | 1 | 77183.60 | 2011-01-18 | -325 | 1 | 1 | 5 | 115 | Cannot Lose Them | 115 | Non Loyal | 74125.000000 | 77183.000000 | 74215 | 77183.6 |
| 1 | 12347.0 | 2 | 182 | 4310.00 | 2010-12-07 | -2 | 5 | 5 | 5 | 555 | Champions | 555 | Non Loyal | 13.505495 | 23.681319 | 12 | 25.2 |
| 2 | 12348.0 | 75 | 27 | 1595.64 | 2010-12-16 | -75 | 2 | 2 | 4 | 224 | At Risk | 224 | Non Loyal | 68.925926 | 59.097778 | 72 | 39.6 |
| 3 | 12349.0 | 18 | 73 | 1757.55 | 2011-11-21 | -18 | 4 | 4 | 4 | 444 | Loyal | 444 | Loyal | 8.643836 | 24.076027 | 2 | 15.0 |
| 4 | 12350.0 | 310 | 17 | 334.40 | 2011-02-02 | -310 | 1 | 2 | 2 | 122 | Hibernating customers | 122 | Non Loyal | 11.588235 | 19.670588 | 12 | 25.2 |

### 3️⃣ Visualization  

<details>
 <summary><strong>Histogram for R, F, M scores:</strong></summary>

 ```python
 # Histograms for R, F, and M scores
 fig, axes = plt.subplots(1, 3, figsize=(18, 6))
 
 # Convert R, F, and M columns to integer type for correct ordering
 df_RFM_final['R_int'] = df_RFM_final['R'].astype(int)
 df_RFM_final['F_int'] = df_RFM_final['F'].astype(int)
 df_RFM_final['M_int'] = df_RFM_final['M'].astype(int)
 
 
 sns.histplot(data=df_RFM_final, x='R_int', ax=axes[0], kde=True, discrete=True)
 axes[0].set_title('Distribution of Recency (R) Scores')
 axes[0].set_xlabel('Recency Score')
 axes[0].set_ylabel('Number of Customers')
 axes[0].set_xticks(range(1, 6)) # Set explicit tick locations
 
 sns.histplot(data=df_RFM_final, x='F_int', ax=axes[1], kde=True, discrete=True)
 axes[1].set_title('Distribution of Frequency (F) Scores')
 axes[1].set_xlabel('Frequency Score')
 axes[1].set_ylabel('Number of Customers')
 axes[1].set_xticks(range(1, 6)) # Set explicit tick locations
 
 sns.histplot(data=df_RFM_final, x='M_int', ax=axes[2], kde=True, discrete=True)
 axes[2].set_title('Distribution of Monetary (M) Scores')
 axes[2].set_xlabel('Monetary Score')
 axes[2].set_ylabel('Number of Customers')
 axes[2].set_xticks(range(1, 6)) # Set explicit tick locations
 
 
 plt.tight_layout()
 plt.show()
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_visualization_R_F_M.png)

➡️ As can be seen from the histogram:  
- **Recency (R):** The chart shows that most customers have high Recency scores (4 and 5), concentrated on the right side of the distribution. This indicates that a majority of customers have made recent purchases. However, there is still a significant portion of customers with low Recency scores (1, 2, or 3), suggesting they haven't purchased in a while.
- **Frequency (F):** The frequency distribution is left-skewed, with most customers having low Frequency scores (1 and 2). This indicates that the majority of customers do not purchase frequently. A small segment of customers with high Frequency scores (4 and 5) represents those who buy very regularly.
- **Monetary (M):** The Monetary distribution is also left-skewed, similar to Frequency. This suggests that most customers have low spending values. Only a small number of customers have high Monetary scores (4 and 5), representing high-value spenders.  

<details>
 <summary><strong>Visualize final dataset with RFM:</strong></summary>

 ```python
 # Visualize spending amount and number of user according to Segment.
 user_by_segment = df_RFM_final[['Segment','CustomerID']].groupby(['Segment']).count().reset_index()
 user_by_segment = user_by_segment.rename(columns = {'CustomerID':'user_volume'})
 user_by_segment['contribution_percent'] = round(user_by_segment['user_volume'] / user_by_segment['user_volume'].sum() * 100)
 user_by_segment['type'] = 'user contribution'
 
 spending_by_segment = df_RFM_final[['Segment','Monetary']].groupby(['Segment']).sum().reset_index()
 spending_by_segment = spending_by_segment.rename(columns = {'Monetary':'spending'})
 spending_by_segment['contribution_percent'] = spending_by_segment['spending'] / spending_by_segment['spending'].sum() * 100
 spending_by_segment['type'] = 'spending contribution'
 
 segment_agg = pd.concat([user_by_segment, spending_by_segment])
 
 plt.figure(figsize=(20, 10))
 sns.barplot(segment_agg, x='Segment', y='contribution_percent', hue='type')
 plt.title='Spending amount and number of user according to Segment'
 
 plt.show()
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_visualization.png)

<details>
 <summary><strong>Visualize for sales trending:</strong></summary>

 ```python
 monthly_sales = df_main.groupby('Month')['Sales'].sum().reset_index()
 
 # Convert Month to datetime for proper sorting
 monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'])
 
 # Sort by month
 monthly_sales = monthly_sales.sort_values('Month')
 
 # Visualize the sales trend over time
 plt.figure(figsize=(12, 6))
 sns.lineplot(data=monthly_sales, x='Month', y='Sales')
 plt.xlabel('Month')
 plt.ylabel('Total Sales')
 plt.xticks(rotation=45)
 plt.tight_layout()
 plt.show()
 ```
</details>

![](https://github.com/longnguyen0102/photo/blob/main/RFM_analysis-retail-python/RFM_analysis-retail-python_visualization_sales_trending.png)

### 4️⃣ Insights and Actions (drawing from both graphs of RFM and sales trending)  

✔️ The **"Champions"** segment is the core revenue driver: The chart shows that the **"Champions"** group contributes the largest share of revenue—over 60%—despite representing only around 18% of the total customer base. This highlights the critical importance of this segment to SuperStore. These are the most frequent, recent, and high-spending customers.  
➡️ **Action:** It is essential to focus on maintaining and enhancing the experience for **"Champions"** to ensure stable and sustainable revenue.

✔️ The **"Loyal"** segment also makes a significant contribution: The **"Loyal"** customers account for approximately 10% of the total customer base and contribute a notable portion of revenue—over 10%.  
➡️ **Action:** This is a high-potential segment that can be nurtured to become future **"Champions"** Targeted initiatives such as personalized offers, loyalty programs, or incentives could encourage them to increase purchase frequency and order value.  

✔️ The **"Potential Loyalist"** segment shows promise but needs activation: The **"Potential Loyalist"** group represents a relatively high share of the customer base (11%) but contributes only around 3.2% of total revenue. This aligns with the typical characteristics of this segment—good Recency and Frequency, but low Monetary value.  
➡️ **Action:** Targeted campaigns should aim to increase spending per transaction for this group in order to convert them into **"Loyal"** or even **"Champions"** over time. Strategies could include personalized upselling, product bundling, or limited-time promotions to encourage higher basket sizes.

✔️ Based on the *sales trending* graph:  
➡️ Quarter fourth is a good time for **upselling**. This is the time that customers will spend more money for preparing for Holiday Season. Upselling programs are focus on increasing average order value instead of discount.  
➡️ Months in early and middle of the year are the time for launching **customer incentive and relation programs**. During these time, the need for buying is low. That is the reason for these programs to step in, they will attract more customers (even new ones) and increase customers' Frequency, like: price discount, buy 1 get 1, voucher for the next buying,...  
➡️ Months before sales increasing (such as September) is the time for **"heat up the market"**. Launching early promotion programs, new products, new collections are not the bad idea.  

## 📌 Key Takeaways:  
✔️ Understand how **RFM analysis** can be used to evaluate customer behavior based on purchase frequency and spending value.  

✔️ **Classify customers** into specific segments using RFM scores, helping identify which segments require enhanced experiences and which should be retained and nurtured to move toward higher-value tiers.  

✔️ Determine the **optimal timing** for launching promotional campaigns and upselling strategies, enabling the business to both retain existing customers and attract new ones.
