import pandas as pd
import ast

# Load the main DataFrame
df = pd.read_csv("books_data_clean.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

# Load the categories DataFrame
categories_df = pd.read_csv("categories.csv")

# Step 1: Convert strings to lists and flatten the 'categories' column
df['categories'] = df['categories'].apply(ast.literal_eval)
all_categories = df['categories'].explode()

# Step 2: Count the frequency of each category
category_counts = all_categories.value_counts()

# Step 3: Select the top 15 categories
top_categories = category_counts.head(15).reset_index()
top_categories.columns = ['category', 'count']

# Step 4: Merge with categories DataFrame to get the correct IDs
top_categories = top_categories.merge(categories_df, on='category', how='left')

# Step 5: Reorder columns to place 'id' first
top_categories = top_categories[['id', 'category', 'count']]

# Step 6: Export the DataFrame to a CSV file
top_categories.to_csv('TopCategories.csv', index=False)

print("TopCategories.csv")
