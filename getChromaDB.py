from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pandas as pd
import shutil
import os

# Load and prepare the new data
df = pd.read_csv("books_data_clean_with_id.csv")
df["Metadata"] = df.apply(
    lambda row: f"Book titled '{row['title']}'. Description: {row['description']}. "
                f"Authors: {row['authors']}. Published by {row['publisher']}. "
                f"Category: {row['categories']}. Rating: {row['ratingsCount']}.", axis=1)
df.to_csv("data_book_metadata.csv", index=False)

# Load the CSV data
loader = CSVLoader(file_path="data_book_metadata.csv", encoding="utf-8")
new_data = loader.load()

# Create the embedding function using the fine-tuned model
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Specify the folder for the database
folder_db = "db"

# Delete the existing database folder if it exists
if os.path.exists(folder_db):
    shutil.rmtree(folder_db)

# Create a new database
db = Chroma.from_documents(new_data, embedding_function, persist_directory=folder_db)

# Persist the new database
db.persist()

print("New data loaded into Chroma with embeddings, old database deleted and new database created.")
