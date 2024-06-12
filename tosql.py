import pandas as pd

from sqlalchemy import create_engine


data = pd.read_csv("books_data_clean_with_id.csv")
data_rating = pd.read_csv("books_rating_clean_with_book_id.csv")

data.rename(columns={"Title": "title"}, inplace=True)
data = data.drop(columns=["Unnamed: 0"])

data_rating.rename(columns={"Title": "title", "Price": "price", "User_id": "user_id", "Id": "goodreads_book_id", "review/summary": "reviewSummary", "review/text": "reviewText", "review/time": "reviewTime", "review/score": 'reviewScore', "review/helpfulness": "reviewHelpfulness"}, inplace=True)
data_rating = data_rating.drop(columns=["Unnamed: 0"])



engine = create_engine("mysql+pymysql://root:@localhost/finder")


data.to_sql('book', con=engine, if_exists='replace', index=False)
data_rating.to_sql('rating', con=engine, if_exists='replace', index=False)

print("selesai")