import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np

# Load data
books = pd.read_csv("books_data_clean_with_id.csv")
ratings = pd.read_csv("books_rating_clean_with_book_id.csv")

# Merge dataframes on book_id
df = pd.merge(ratings, books, left_on='book_id', right_on='id')

# Filter out books with less than the minimum number of ratings
min_ratings_count_threshold = 10
rating_counts = df.groupby('book_id').count()['review/score']
popular_books = rating_counts[rating_counts >= min_ratings_count_threshold].index

# Get final ratings for popular books
final_ratings = df[df['book_id'].isin(popular_books)][['book_id', 'user_id', 'review/score']]

# Save final ratings to a CSV
final_ratings.to_csv('final_ratings.csv', index=False)

# Create the pivot table
pivot_table = final_ratings.pivot_table(index='book_id', columns='user_id', values='review/score')
pivot_table.fillna(0, inplace=True)

# Save the index and columns for later use
pivot_table_index = pivot_table.index
pivot_table_columns = pivot_table.columns

# Convert the pivot table to a csr_matrix
csr_pivot_table = csr_matrix(pivot_table.values)

# Save the csr_matrix in a memory-efficient way
save_npz('pivot_table.npz', csr_pivot_table)
pivot_table_index.to_series().to_csv('pivot_table_index.csv', header=True)
pivot_table_columns.to_series().to_csv('pivot_table_columns.csv', header=True)

print("CSR matrix saved successfully.")
