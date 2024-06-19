# Book Recommendation System

## Description
This project consists of a book recommendation system that uses three algorithms: content-based filtering and two types of collaborative filtering. The content-based algorithm embeds book metadata using the mini all-MiniLM-L6-v2 model and stores them in ChromaDB for similarity searches. The collaborative filtering algorithms recommend books based on user reviews; one using cosine similarity on a review pivot table and the other using a trained RecommenderNet model.

## Prerequisites
- Python 3.9 or 3.10
- Python packages listed in `requirements.txt`
- ngrok or Google Cloud Platform (optional)

## Instructions
1. Ensure Python with all required packages is installed.
2. Clone this repository:
    ```sh
    git clone https://github.com/LitFinder/Recommender.git
    cd Recommender
    ```
3. Download the CSV dataset [here](https://drive.google.com/drive/folders/1soX0jyy1ZrKaT_nmcxFwbw00aA0FSWcE?usp=sharing) (dataset too large for Git).
4. Add the CSV files to the cloned repository location.
5. Run the following scripts to set up the recommendation models:
    ```sh
    python getColabUser.py
    python getPivotTable.py
    python getChromaDB.py
    ```
6. Give the `restart_server.sh` script executable permissions:
    ```sh
    chmod +x restart_server.sh
    ```
7. Execute the `restart_server.sh` script to start the server:
    ```sh
    ./restart_server.sh
    ```
8. The server should run at [http://0.0.0.0:8000/docs](http://0.0.0.0:8000/docs) or [http://localhost:8000/docs](http://localhost:8000/docs).

### Optional
- Test the API using ngrok. If ngrok is set up, run:
    ```sh
    ngrok http 8000
    ```

## Known Issues
- If the ColabUser recommender doesn't work for a new user, ensure the line to run `getColabUserRetry.py` in `api.py` is uncommented. (This line is commented out by default because the model requires retraining for new users, which is compute-intensive and not feasible within our current GCP limitations.)
```
