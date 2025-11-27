import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

#Data Inspection

df = pd.read_csv('movies_data.csv')
print(df.head())
print(df.tail())
print(df.info())

for col in df.columns:
    print(f"Column {col}:")
    print(f"Number of unique values: {len(df[col].value_counts())}")
    print(f"Number of null values: {df[col].isnull().sum()}")
    print("------------------------------")

duplicates = df.duplicated().sum()
print(f"\n Number of duplicates: {duplicates} ")

numerical_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
object_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features = ['rating','language','gernes']

print(f"Numerical Data: {numerical_features}")
print(f"\nObject Data: {object_features}")

print(df[numerical_features].describe())

print("-------Inspection Summary-------")

# 1. Missing values
print("1. MISSING VALUES:")
print(df.isnull().sum())
print("\nMovie with missing genre:")
print(df[df['genres'].isnull()])

# 2. Duplicates
print("\n2. DUPLICATES:")
print(f"Number of duplicates: {df.duplicated().sum()}")
print("\nDuplicate rows:")
print(df[df.duplicated(keep=False)])

# 3. Zero values (invalid data)
print("\n3. ZERO/INVALID VALUES:")
print(f"Movies with runtime = 0: {len(df[df['runtime'] == 0])}")
print(df[df['runtime'] == 0][['title', 'runtime']])
print(f"\nMovies with budget = 0: {len(df[df['budget'] == 0])}")
print(f"Movies with revenue = 0: {len(df[df['revenue'] == 0])}")
print(f"Movies with rating = 0: {len(df[df['rating'] == 0])}")
print(f"Movies with votes = 0: {len(df[df['votes'] == 0])}")

# 4. Outliers
print("\n4. OUTLIERS:")
print("\nTop 5 highest votes:")
print(df.nlargest(5, 'votes')[['title', 'votes']])
print("\nTop 5 highest revenue:")
print(df.nlargest(5, 'revenue')[['title', 'revenue']])
print("\nTop 5 highest popularity:")
print(df.nlargest(5, 'popularity')[['title', 'popularity']])

# 5. Date format issues
print("\n5. DATE FORMAT:")
print("Sample release dates:")
print(df['release_date'].head(10))

# 6. Consistency checks
print("\n6. CONSISTENCY:")
print(f"Unique titles: {df['title'].nunique()}")
print(f"Total movies: {len(df)}")
print("\nLanguage distribution:")
print(df['language'].value_counts())
print("\nGenres with issues:")
print(df[df['genres'] == ''][['title', 'genres']])