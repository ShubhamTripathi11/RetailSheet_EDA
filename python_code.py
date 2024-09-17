#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Importing required libraries
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics as stc 
from scipy.stats import norm
import scipy.stats.stats


# In[73]:


# Load the CSV data
csv_file_path = 'C:\\Users\\91983\\Desktop\\EDA Python\\cleaned_retail_price.csv'
sales= pd.read_csv(csv_file_path)


# In[59]:


sales.head()


# In[63]:


# Check for null values
null_values = sales.isnull().sum()
# Print the count of null values for each column
print("Null values in each column:")
print(null_values)


# In[64]:


#checking the datatypes in the dataframe
sales.info()


# In[79]:


# Print the categories
product_categories = sales['product_category_name'].unique()

print("Categories of products sold:")
for category in product_categories:
    print(category)


# In[69]:


# Top 20 products by sales volume

products = sales [["product_category_name", "product_id", "volume"]].copy()
products = products.sort_values (by = "volume", ascending = False)
print("Top 20 Products: ")
products.drop_duplicates().head(20)


# In[80]:


#List of all product sold
products_sold = sales[['product_id', 'product_category_name']].drop_duplicates()

# Print the list of all products sold along with their IDs
print("List of all products sold:")
print(products_sold)


# In[82]:


# Display descriptive statistics for sales data
sales_statistics = sales[['qty', 'total_price', 'freight_price', 'unit_price']].describe()

# Print descriptive statistics
print("Sales Data Descriptive Statistics:")
print(sales_statistics)


# In[83]:


new_sales = sales [["product_category_name", "product_id", "month_year", "unit_price", "qty","total_price"]].copy()
new_sales


# In[84]:


# Assigning product category name as column attributes 
new_sales = new_sales.transpose()
new_sales.rename(columns=new_sales.iloc[0], inplace = True)
new_sales.drop(new_sales.index [0], inplace = True)
new_sales


# In[86]:


#Elasticity to be calculated across following categories 
all_products = new_sales.columns.drop_duplicates()
print("Number of product categories",len(all_products), "\n")
print("Product Categories:", *all_products, sep = "\n")

print(f"\nThere are a total of {len(all_products)} categories across which Elasticity must be calculated.")


# In[87]:


# Determine elasticity by elasticity coeffecient
def els_coef (value):
    if value >1:
        result = "Elastic" 
    elif value == 1:
        result = "Unitarily elastic"   
    elif value == 0:
        result = "Perfectly inelastic"
    elif value<1 and value!=0: 
        result = "Inelastic"
    return result


# In[88]:


bed = new_sales["bed_bath_table"].copy().transpose()


# In[89]:


# Grouping sub-categories under Bed & Bath category
catg = bed.groupby("product_id")


# In[90]:


# Storing the grouped dataframes in a dictionary
bed_tables ={}
for bed_id, bed_df in catg:
    bed_tables [bed_id] = bed_df


# In[91]:


#aggregating quantity for some untt prices 
for sub in bed_tables:
    bed_tables[sub] = bed_tables[sub].groupby("unit_price")["qty"].mean().reset_index()


# In[92]:


#Calculating elasticity for all sub-categories:
for sub in bed_tables:
    bed_tables[sub]["%change_qty"] = bed_tables[sub]["qty"].pct_change()*100
    bed_tables[sub]["%change_price"] = bed_tables[sub]["unit_price"].pct_change()*100
    bed_tables[sub]["elasticity"] = round(bed_tables[sub]["%change_qty"]/bed_tables[sub]["%change_price"],2)
    bed_tables[sub].dropna(inplace = True)
    print(sub, "\n", bed_tables[sub], "\n")


# In[93]:


for sub in bed_tables:
    print(f'corr_{sub} : {scipy.stats.pearsonr(bed_tables[sub]["unit_price"], bed_tables[sub]["qty"])[0]:.2}')


# In[94]:


#ALL elasticity values for Bed and Bath Category
bed_elasticity = sum([list(bed_tables[sub] ["elasticity"]) for sub in bed_tables], [])


# In[95]:


#Elasticity central tendency values for Bed & Bath
print("Elasticity central tendency values for Bed & Bath")
bed_els_mean = stc.fmean(bed_elasticity)
bed_els_median = stc.median (bed_elasticity) 
bed_els_mode = stc.mode(bed_elasticity)
bed_els_std = stc.stdev (bed_elasticity)

print(f"Mean ={bed_els_mean:.2f}")
print("Median = ",bed_els_median) 
print("Mode = ", bed_els_mode)
print(f"Standard Deviation = {bed_els_std:.2f}")
print(f"Variance = {stc.variance(bed_elasticity):.2f}")


# In[ ]:


def calculate_elasticity(sales, product_category):
    # Filter data for the specified product category
    category_data = sales[sales['product_category_name'] == product_category]
    
    # Calculate the average price and quantity
    avg_price = category_data['total_price'].mean()
    avg_qty = category_data['qty'].mean()
    
    # Calculate the price elasticity of demand
    elasticity = (avg_qty / avg_price) * (category_data['total_price'].mean() / category_data['qty'].mean())
    
    return elasticity


# In[ ]:


product_categories = sales['product_category_name'].unique()


# In[ ]:





# In[115]:


# Define a function to calculate elasticity
def calculate_elasticity(df, product_category):
    # Filter data for the specified product category
    category_data = df[df['product_category_name'] == product_category]
    
    # Calculate the average price and quantity
    avg_price = category_data['total_price'].mean()
    avg_qty = category_data['qty'].mean()
    
    # Calculate the price elasticity of demand
    elasticity = (avg_qty / avg_price) * (category_data['total_price'].mean() / category_data['qty'].mean())
    
    return elasticity


# In[117]:


# Get unique product categories
product_categories = sales['product_category_name'].unique()

# Calculate elasticity for each product category
elasticity_results = {}
for category in product_categories:
    elasticity = calculate_elasticity(sales, category)
    elasticity_results[category] = elasticity

# Print elasticity results
for category, elasticity in elasticity_results.items():
    print(f"Elasticity for {category}: {elasticity}")


# In[122]:


sales_volume = sales.groupby('product_category_name')['qty'].sum().sort_values(ascending=False)


# In[123]:


colors = plt.cm.viridis(np.linspace(0, 1, len(sales_volume)))


# In[124]:


plt.figure(figsize=(10, 6))
sales_volume.plot(kind='bar', color=colors)
plt.title('Product Categories by Sales Volume')
plt.xlabel('Product Category')
plt.ylabel('Sales Volume')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[125]:


plt.figure(figsize=(8, 6))
plt.hist(sales['product_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Product Score Distribution')
plt.xlabel('Product Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[104]:


# Price Distribution
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(sales['unit_price'])
ax.axvline(stc.fmean(sales['unit_price']), color = 'r')
ax.axvline(stc.median(sales['unit_price']), color = 'yellow')
ax.legend(["Mean", "Median"])
ax.set_title("Retail Price Distribution")
ax.set_xlabel("Unit Price")


# In[105]:


# Regresssion Analysis for Unit Price vs Quantity sold

y = sales['unit_price']
x = (sales['qty'])


fig, ax = plt.subplots(figsize=(8, 4))
sns.regplot(sales, y = y,x = x, line_kws={"color": "orange"})
ax.set_title("Unit Price vs Quantity Analysis")
ax.set_xlabel("Quantity")
ax.set_ylabel("Unit Price")
plt.tight_layout()


# In[106]:


#Unit price Comparision with Competitors
prices = sales [["unit_price", "comp_1", "comp_2", "comp_3"]].copy()

fig, ax = plt.subplots(figsize=(14,4))
sns.boxplot(prices, orient = 'h', width= 0.8)

ax.set_title("Unit price Distribution and Comparision with Competitors")
ax.set_xlabel('Unit price (INR)')
ax.set_ylabel('Company')

plt.tight_layout()

print("Unit Price Comparision with Median", end = "")
prices.median()


# In[126]:


plt.figure(figsize=(12, 6))
for category, data in sales.groupby('product_category_name'):
    plt.hist(data['qty'], bins=100, alpha=0.5, label=category)

plt.title('Quantity Sold Distribution by Product Category')
plt.xlabel('Quantity Sold')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# In[127]:


category_sales = sales.groupby('product_category_name')['total_price'].sum()

# Generating a color palette
colors = plt.cm.viridis(np.linspace(0, 1, len(category_sales)))

# Plotting
plt.figure(figsize=(12, 8))
category_sales.plot(kind='bar', color=colors)
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.grid(axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[128]:


sales['month_year'] = pd.to_datetime(sales['month_year'])

# Extract month and year from the "month_year" column
sales['month'] = sales['month_year'].dt.month
sales['year'] = sales['month_year'].dt.year

# Calculate revenue
sales['revenue'] =sales['qty'] * sales['unit_price']

# Create a pivot table to sum revenue by product category, month, and year
pivot_table =sales.pivot_table(index=['year', 'month'], columns='product_category_name', values='revenue', aggfunc='sum')

# Plotting
plt.figure(figsize=(12, 8))

# Plot each product category separately
for category in pivot_table.columns:
    pivot_table[category].plot(kind='line', marker='o', linewidth=2, label=category)

plt.title('Sum of Monthly Revenue by Product Category')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[110]:


fig, ax = plt.subplots(figsize=(10,3))
sns.boxplot(sales[['product_score', 'ps1', 'ps2', 'ps3']], width=0.7, orient='h')
ax.set_title('Product score Distribution and Comparision with competitors',)
ax.set_xlabel("Product Score (0-5)")
ax.set_ylabel("Company")

plt.tight_layout()
print("Product Score Comparision with Median", end = "") 
sales [['product_score', 'ps1', 'ps2', 'ps3']].median()


# In[111]:


#Data Visualization: Price elasticity for all subcategories

#creating subplots
fig, axes = plt.subplots(3,2, sharex = True, sharey = True, figsize = (10,10))


#storing the table in a temp df
bed_tables_temp =bed_tables.copy()

#creating a subplot to add common axes and title
fig.add_subplot(111, frameon = False)
plt.grid(False)
plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)

#plotting subplots
for i, ax in enumerate (axes.flatten()):
    for sub in bed_tables_temp:
        ax.scatter(bed_tables_temp[sub]["qty"], bed_tables_temp[sub]["unit_price"], color = "r")
        ax.set_title(f"{sub}", fontsize = 12, fontfamily = "serif")
        bed_tables_temp.pop(sub)
        break
        
plt.xlabel("Demand Quantity", fontfamily = "serif", fontsize = 10)
plt.ylabel("Price", fontfamily = "serif", fontsize = 10)
fig.suptitle("Price vs Sales Quantity Analysis by Sub-category\n\n", fontfamily = "serif", fontsize= 13)
plt.show()


# In[129]:


#Data Visualization: Price elasticity for Bed&Bath
for sub in bed_tables:
    plt.scatter (bed_tables[sub]["qty"],bed_tables[sub]["unit_price"])

    
plt.xlabel("Demand Quantity", fontfamily = "serif", fontsize = 10)
plt.ylabel("Price", fontfamily = "serif", fontsize = 10)
plt.title("Bed & Bath : Price vs Sales Quantity Analysis", fontsize = 12, fontfamily = "serif")
plt.legend(["Elasticity", "Elasticity", "Elasticity", "Elasticity", "Elasticity"])
plt.show()



# In[143]:


#Data Visualization: Elasticity Distribution
plt.figure(figsize = (6,4))

for sub in bed_tables:
    sns.histplot(bed_tables[sub]["elasticity"], kde = True)

plt.xlabel("Elasticity", fontfamily = "serif", fontsize = 20)
plt.ylabel("Frequency", fontfamily = "serif", fontsize = 20)
plt.title("Bed & Bath Elasticity Distribution: All Sub-categories", fontsize = 24, fontfamily = "serif")

plt.show()


# In[ ]:





# In[ ]:




