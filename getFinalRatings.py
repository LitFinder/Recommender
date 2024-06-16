import pandas as pd

books = pd.read_csv("books_data_clean_with_id.csv")
ratings = pd.read_csv("books_rating_clean_with_book_id.csv")

df = pd.merge(ratings, books, left_on='book_id', right_on='id')

min_ratings_count_threshold=10
rating_counts= df.groupby('book_id').count()['review/score']
popular_books = rating_counts[rating_counts >= min_ratings_count_threshold].index

final_ratings =  df[df['book_id'].isin(popular_books)]

final_ratings = df[df['book_id'].isin(popular_books)][['book_id', 'user_id', 'review/score']]

final_ratings.to_csv('final_ratings.csv')