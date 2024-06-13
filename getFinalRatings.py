import pandas as pd

books = pd.read_csv("books_data_clean.csv")
ratings = pd.read_csv("books_rating_clean.csv")
books.drop(columns=["Unnamed: 0"], inplace=True)
ratings.drop(columns=["Unnamed: 0"], inplace=True)

df = pd.merge(books, ratings, on='Title', how='inner')

min_ratings_count_threshold=50
rating_counts= df.groupby('Title').count()['review/score']
popular_books = rating_counts[rating_counts >= min_ratings_count_threshold].index

final_ratings =  df[df['Title'].isin(popular_books)]

final_ratings = df[df['Title'].isin(popular_books)][['Title', 'User_id', 'review/score']]

final_ratings.to_csv('final_ratings.csv')