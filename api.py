from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from typing import List, Dict, Optional
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import tensorflow as tf  # Make sure TensorFlow is imported to use the model
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import logging
from tensorflow import keras
from scipy.sparse import load_npz
from fastapi import Depends


logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level (e.g., INFO, DEBUG)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    # Load your data
    books = pd.read_csv("books_data_clean_with_id.csv")
    ratings_df  = pd.read_csv("books_rating_clean_with_book_id.csv")
# final_ratings = pd.read_csv("final_ratings.csv")
    # pivot_table = pd.read_csv("PivotTable.csv")
    merged_df = pd.merge(ratings_df, books, left_on='book_id', right_on='id')

    persist_directory = "db"
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Map user ID to a "user vector" via an embedding matrix
    user_ids = merged_df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    # Map books ID to a "books vector" via an embedding matrix
    book_ids = merged_df["book_id"].unique().tolist()
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    book_encoded2book = {i: x for i, x in enumerate(book_ids)}
    merged_df["user"] = merged_df["user_id"].map(user2user_encoded)
    merged_df["book"] = merged_df["book_id"].map(book2book_encoded)
    num_users = len(user2user_encoded)
    num_books = len(book_encoded2book)
    merged_df['rating'] = merged_df['review/score'].values.astype(np.float32)

    # Create pivot table and calculate similarity score
    # pivot_table = final_ratings.pivot_table(index='book_id', columns='user_id', values='review/score')
    # pivot_table.fillna(0, inplace=True)
    # Load the csr_matrix
    csr_pivot_table = load_npz('pivot_table.npz')
    pivot_table_index = pd.read_csv('pivot_table_index.csv', index_col=0)
    pivot_table_columns = pd.read_csv('pivot_table_columns.csv', index_col=0)
    # Convert the CSR matrix back to a dense array
    dense_pivot_table = csr_pivot_table.toarray()
    # Convert the dense array back to a pandas DataFrame
    pivot_table = pd.DataFrame(dense_pivot_table, index=pivot_table_index.squeeze(), columns=pivot_table_columns.squeeze())
    similarity_score = cosine_similarity(pivot_table)

    # Load the pre-trained recommendation model
    model = keras.models.load_model("Colab_User")
    
    # Return necessary data as a dictionaryz
    return {
        "books": books,
        "ratings_df": ratings_df,
        # "final_ratings": final_ratings,
        "merged_df": merged_df,
        "persist_directory": persist_directory,
        "embedding": embedding,
        "vectordb": vectordb,
        "user2user_encoded": user2user_encoded,
        "userencoded2user": userencoded2user,
        "book2book_encoded": book2book_encoded,
        "book_encoded2book": book_encoded2book,
        "num_users": num_users,
        "num_books": num_books,
        "pivot_table": pivot_table,
        "similarity_score": similarity_score,
        "model": model
    }
    


app = FastAPI()

# ------------------------- Load Data -------------------------
# Load data when FastAPI starts
@app.on_event("startup")
async def startup_event():
    app.state.data = load_data()  # Store loaded data in app state


# Example endpoint to reload data (for demonstration)
@app.get("/reload-data/")
async def reload_data():
    app.state.data = load_data()  # Reload data
    return {"message": "Data reloaded successfully"}

# Endpoint to restart the API
@app.post("/restart-api/")
async def restart_api():
    try:
        subprocess.Popen(["./restart_server.sh"])
        return {"message": "API is restarting..."}
    except Exception as e:
        logger.error(f"Failed to restart the API: {e}")
        raise HTTPException(status_code=500, detail="Failed to restart the API")


# ------------------------- APScheduler Setup -------------------------
scheduler = BackgroundScheduler()

# Function to be executed every 3 days
def scheduled_task():
    logger.info("Scheduled task is running...")
    try:
        subprocess.run(["python", "getChromaDB.py"], check=True)
        subprocess.run(["python", "getPivotTable.py"], check=True)
        subprocess.run(["python", "getColabUser.py"], check=True)
        restart_api()
        logger.info("Scheduled task completed successfully.")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Scheduled task failed: {e}")

# Schedule the task to run every 3 days
scheduler.add_job(scheduled_task, 'interval', days=3)
scheduler.start()

# Manually trigger task
@app.post("/trigger_task/")
async def trigger_task():
    scheduled_task()
    return {"message": "Scheduled task triggered manually"}

# ------------------------- Content Based Filtering Recommendation -------------------------
class RecommendationRequest(BaseModel):
    id_book: Optional[List[int]] = None

@app.post("/recommendation/")
async def recommendation(request: RecommendationRequest):
    vectordb = app.state.data["vectordb"]
    # Fungsi untuk mendapatkan buku dengan rating tertinggi
    def get_top_rated_books(n: int):
        df = pd.read_csv('books_data_clean_with_id.csv') # Ubah sesuai dengan path file Anda
        df_sorted = df.sort_values(by='ratingsCount', ascending=False)
        top_100 = df_sorted.head(n)
        return top_100.index.tolist()

    if request.id_book:
        k_recommendation = round(100 / len(request.id_book))
    
        all_recommendation = []
        for book_recomen in request.id_book:
            book_recomen = str(book_recomen)
            similarity_questions = vectordb.similarity_search(book_recomen, k=k_recommendation)

            docs = similarity_questions[1::]
        
            recommendation_list = []
            for doc in docs:
                row_value = doc.metadata.get("row")
                recommendation_list.append(row_value)

            all_recommendation.extend(recommendation_list)
        
        recommendation_indices = list(np.array(all_recommendation).flatten().tolist())
    else:
        recommendation_indices = get_top_rated_books(100)

    return {"recommendations": recommendation_indices}

# ------------------------- Colabortive Filtering Recommendation on Books -------------------------
@app.post("/colabBook/")
async def recommend(id_book: int = Query(...), amount: int = Query(...)):
    books = app.state.data["books"]
    pivot_table = app.state.data["pivot_table"]
    similarity_score = app.state.data["similarity_score"]
    try:
        book_getId = books.loc[books.id == id_book, 'id'].values[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Book ID not found")
    
    if book_getId not in pivot_table.index:
        raise HTTPException(status_code=404, detail="Book title not found in pivot table")

    index = np.where(pivot_table.index == book_getId)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:amount + 1]
    
    recommended_books = []
    for idx, _ in similar_books:
        recommended_book = books[books["id"] == pivot_table.index[idx]]
        recommended_books.append(recommended_book)
        
    recommendations = []
    for index, row in pd.concat(recommended_books).iterrows():
        recommendations.append(row["id"])
    
    return {
        # "user_id": user_id,
        # "top_books_user": top_books,
        "recommendations": recommendations
    }

# ------------------------- Colabortive Filtering Recommendation on Users -------------------------
def getColabUserData(): 
    books = app.state.data["books"]
    book2book_encoded = app.state.data["book2book_encoded"]
    user2user_encoded = app.state.data["user2user_encoded"]
    book_encoded2book = app.state.data["book_encoded2book"]
    merged_df = app.state.data["merged_df"]
    model = app.state.data["model"]
    
    return {
        'books': books,
        'book2book_encoded' : book2book_encoded,
        'user2user_encoded' : user2user_encoded,
        'book_encoded2book' : book_encoded2book,
        'merged_df' : merged_df,
        'model' : model,
    }
    
async def get_colab_user_data():
    return getColabUserData()


@app.post("/colabUser/")
async def recommend_for_user(user_id: str = Query(...), amount: int = Query(...), data: dict = Depends(get_colab_user_data)):
    books_watched_by_user = data['merged_df'][data['merged_df']['user_id'] == user_id]
    books_not_watched = data['books'][~data['books']['id'].isin(books_watched_by_user['book_id'].values)]['id']

    books_not_watched = list(set(books_not_watched).intersection(set(data['book2book_encoded'].keys())))

    books_not_watched = [[data['book2book_encoded'].get(x)] for x in books_not_watched]

    user_encoder = data['user2user_encoded'].get(user_id)

    if user_encoder is None:
        raise HTTPException(status_code=404, detail="User ID not found")

    user_book_array = np.hstack(
        ([[user_encoder]] * len(books_not_watched), books_not_watched)
    )

    ratings = data['model'].predict(user_book_array).flatten()
    top_ratings_indices = ratings.argsort()[-amount:][::-1]
    recommended_book_ids = [
        data['book_encoded2book'].get(books_not_watched[x][0]) for x in top_ratings_indices
    ]

    recommended_books = data['books'][data['books']["id"].isin(recommended_book_ids)]
    recommendations = []
    for index, row in recommended_books.iterrows():
        recommendations.append(row["id"])
    return {
        "recommendations": recommendations
    }
    
    
# ------------------------- Add New Rating Data -------------------------
class NewRating(BaseModel):
    id: int
    book_id: int
    price: float
    user_id: str
    profileName: str
    review_helpfulness: str
    review_score: float
    review_time: int
    review_summary: str
    review_text: str

@app.post("/add_rating/")
async def add_rating(new_rating: NewRating, background_tasks: BackgroundTasks):
    try:
        # Create a new DataFrame from the incoming data
        new_data = pd.DataFrame([{
            "id": new_rating.id,  # Use `book_id` from the NewRating instance
            "book_id": new_rating.book_id,  # Use `book_id` from the NewRating instance
            "price": new_rating.price,
            "user_id": new_rating.user_id,
            "profileName": new_rating.profileName,
            "review/helpfulness": new_rating.review_helpfulness,
            "review/score": new_rating.review_score,
            "review/time": new_rating.review_time,
            "review/summary": new_rating.review_summary,
            "review/text": new_rating.review_text
        }])

        # Append the new data to the existing in-memory DataFrame
        ratings_df = app.state.data["ratings_df"]
        app.state.data["ratings_df"] = pd.concat([ratings_df, new_data], ignore_index=True)

        # Schedule the task to save the updated DataFrame to the CSV file
        background_tasks.add_task(save_ratings_to_csv)
        

        return {"message": "Rating added successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_ratings_to_csv():
    ratings_df = app.state.data["ratings_df"]
    ratings_df.to_csv("books_rating_clean_with_book_id.csv", index=False)
    # Call function to update app state data after saving to CSV
    update_app_data()

def update_app_data():
    # Reload necessary data after updating ratings_df
    books = pd.read_csv("books_data_clean_with_id.csv")
    ratings_df = pd.read_csv("books_rating_clean_with_book_id.csv")
    merged_df = pd.merge(ratings_df, books, left_on='book_id', right_on='id')

    # Update embeddings, vectors, encodings, etc.
    # Example: embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Example: vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Update user and book encodings
    user_ids = merged_df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}

    book_ids = merged_df["book_id"].unique().tolist()
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    book_encoded2book = {i: x for i, x in enumerate(book_ids)}
    merged_df["user"] = merged_df["user_id"].map(user2user_encoded)
    merged_df["book"] = merged_df["book_id"].map(book2book_encoded)

    # Update other data structures like pivot_table, similarity_score, model, etc.
    # Example: similarity_score = cosine_similarity(pivot_table)

    # Update app state data
    app.state.data["books"] = books
    app.state.data["ratings_df"] = ratings_df
    app.state.data["merged_df"] = merged_df
    app.state.data["user2user_encoded"] = user2user_encoded
    app.state.data["userencoded2user"] = userencoded2user
    app.state.data["book2book_encoded"] = book2book_encoded
    app.state.data["book_encoded2book"] = book_encoded2book
    # subprocess.run(["python", "getColabUserRetry.py"], check=True)
    # app.state.data["model"] = keras.models.load_model("Colab_User")
    

# Endpoint for root
@app.get("/")
async def read_root():
    return {"message": "Hello World"}


# Ensure the scheduler shuts down properly when the app stops
@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()