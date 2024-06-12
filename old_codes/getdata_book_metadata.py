import pandas as pd

df = pd.read_csv("books_data_clean.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()
df["Metadata"] = df.apply(
    lambda row: f"Book titled '{row['Title']}'. Description: {row['description']}. "
                f"Authors: {row['authors']}. Published by {row['publisher']}. "
                f"Category: {row['categories']}. Rating: {row['ratingsCount']}.", axis=1)

df.to_csv("data_book_metadata.csv", index=False)