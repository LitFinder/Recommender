import pandas as pd
import ast

# Load the DataFrame
df = pd.read_csv("books_data_clean.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

# Step 1: Extract and flatten the 'categories' column
all_categories = df['categories'].apply(ast.literal_eval).explode().unique()

# Step 2: Remove duplicates and ensure each row has only one category
unique_categories = pd.DataFrame(all_categories, columns=['category'])
unique_categories['id'] = range(1, len(unique_categories) + 1)

# Step 3: Reorder columns (optional but recommended)
unique_categories = unique_categories[['id', 'category']]

# Step 3: Export the DataFrame to a CSV file
unique_categories.to_csv('categories.csv', index=False)

print("Categories.csv'")
