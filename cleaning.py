import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("movies_data.csv")

# Data before cleaning:

print("Before Cleaning:")
print(df.shape)  
print(df.isnull().sum())
print(df.describe())
print("\nDuplicates before:", df.duplicated().sum())  
print("Rows with runtime=0:", len(df[df['runtime'] == 0]))  
print("Rows with budget=0:", len(df[df['budget'] == 0]))  
print("Rows with revenue=0:", len(df[df['revenue'] == 0]))  
print("Rows with votes=0:", len(df[df['votes'] == 0]))  

# Handling duplicates:

df = df.drop_duplicates()
df = df.dropna(subset=['genres', 'release_date'])
cols_to_fix = ["budget", "revenue", "runtime",  "votes",'popularity']
for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)


print("\n=== Handling Outliers using IQR method ===")

outlier_columns = ['budget', 'revenue', 'votes', 'popularity']

for col in outlier_columns:
    valid = df[col].dropna()
    Q1 = valid.quantile(0.25)
    Q3 = valid.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    median_val = valid.median()
    
    outliers_count = df[(df[col] < lower) | (df[col] > upper)][col].count()
    
    print(f"{col:<10} → {outliers_count:>3} outliers replaced with {median_val:,.0f}")
    df[col] = df[col].mask((df[col] < lower) | (df[col] > upper), median_val)


# Handling data format:
   
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df[cols_to_fix] = df[cols_to_fix].transform(lambda x: x.fillna(x.median()))
df['genres'] = df['genres'].str.strip().str.replace(' ,', ',').str.replace(', ', ', ')
df['title'] = df['title'].str.strip()
df['title'] = df['title'].str.replace(r'[^\w\s\-\':,]', '', regex=True)
lang_map = {'en':'English','zh':'Chinese','ja':'Japanese','gl':'Galician','lt':'Lithuanian','ml':'Malayalam','nl':'Dutch','is':'Icelandic','es':'Spanish','fr':'French','ko':'Korean','tl':'Tagalog','kn':'Kannada','ta':'Tamil','hi':'Hindi','te':'Telugu','it':'Italian','pl':'Polish','no':'Norwegian','lv':'Latvian','sv':'Swedish','fi':'Finnish','pt':'Portuguese','ca':'Catalan','cn':'Cantonese','th':'Thai','tr':'Turkish','ml':'India','de':'German','id':'Indonesian','sr':'Serbian','el':'Greek','fa':'Farci','mi':'Māori','ru':'Russian','ni':'Dutch','he':'Hebrew','da':'Danish','hu':'Hungarian','ar':'Arabic','ro':'Romanian','cs':'Czech'}
df['language'] = df['language'].map(lang_map).fillna(df['language'])

# Data after cleaning: 

print("\nAfter Cleaning:")
print(df.shape)  
print(df.isnull().sum())  
print(df.describe())
print("\nDuplicates after:", df.duplicated().sum())  
print("Rows with runtime=0:", len(df[df['runtime'] == 0]))  
print("Rows with budget=0:", len(df[df['budget'] == 0]))  
print("Rows with revenue=0:", len(df[df['revenue'] == 0]))  
print("Rows with votes=0:", len(df[df['votes'] == 0]))  

df.to_csv('cleaned_Data.csv', index=False)
