import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import pickle

def save_mappings(user2user_encoded, book2book_encoded, filename='mappings.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((user2user_encoded, book2book_encoded), f)

def load_mappings(filename='mappings.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def update_mappings(merged_df, user2user_encoded, book2book_encoded):
    new_user_ids = merged_df["user_id"].unique().tolist()
    new_book_ids = merged_df["book_id"].unique().tolist()

    for user_id in new_user_ids:
        if user_id not in user2user_encoded:
            user2user_encoded[user_id] = len(user2user_encoded)
    
    for book_id in new_book_ids:
        if book_id not in book2book_encoded:
            book2book_encoded[book_id] = len(book2book_encoded)
    
    return user2user_encoded, book2book_encoded

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_books, embedding_size, dropout_rate=0.2, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(
            num_books,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.book_bias = layers.Embedding(num_books, 1)
        
        self.dropout = layers.Dropout(dropout_rate)
        self.batch_norm = layers.BatchNormalization()
    
    @tf.function 
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = tf.tensordot(user_vector, book_vector, 2)
        # Add all the components (including bias)
        x = dot_user_book + user_bias + book_bias
        
        x = self.dropout(x)
        x = self.batch_norm(x)
        
        # The sigmoid activation forces the rating to be between 0 and 1
        return tf.nn.sigmoid(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_books": self.num_books,
            "embedding_size": self.embedding_size,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def prepare_data():
    # Load data
    book_df = pd.read_csv('books_data_clean_with_id.csv')
    rating_df = pd.read_csv('books_rating_clean_with_book_id.csv')
    merged_df = pd.merge(rating_df, book_df, left_on='book_id', right_on='id')
    return merged_df

def create_mappings(merged_df):
    # Map user ID to a "user vector" via an embedding matrix
    user_ids = merged_df["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}

    # Map books ID to a "books vector" via an embedding matrix
    book_ids = merged_df["book_id"].unique().tolist()
    book2book_encoded = {x: i for i, x in enumerate(book_ids)}
    book_encoded2book = {i: x for i, x in enumerate(book_ids)}

    return user2user_encoded, book2book_encoded

def preprocess_data(merged_df, user2user_encoded, book2book_encoded):
    merged_df["user"] = merged_df["user_id"].map(user2user_encoded)
    merged_df["book"] = merged_df["book_id"].map(book2book_encoded)
    num_users = len(user2user_encoded)
    num_books = len(book2book_encoded)
    merged_df['rating'] = merged_df['review/score'].values.astype(np.float32)

    # min and max ratings will be used to normalize the ratings later
    min_rating = min(merged_df["review/score"])
    max_rating = max(merged_df["review/score"])

    print(f"Number of users: {num_users}, Number of books: {num_books}, Min Rating: {min_rating}, Max Rating: {max_rating}")

    # Shuffle data
    merged_df = merged_df.sample(frac=1, random_state=42)
    x = merged_df[["user", "book"]].values

    # Normalizing the targets between 0 and 1. Makes it easy to train.
    y = merged_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    # Split data into training and validation sets
    train_indices = int(0.9 * merged_df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    return x_train, x_val, y_train, y_val, num_users, num_books

def train_model(x_train, y_train, x_val, y_val, num_users, num_books, embedding_size=64):
    model = RecommenderNet(num_users, num_books, embedding_size)

    # Compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mse', 'accuracy']
    )

    # Train the model for a single step
    model.train_on_batch(x_train, y_train)

    return model

def update_model_for_new_data(model, merged_df, old_num_users, old_num_books, embedding_size=64):
    # Save the current embeddings and biases
    old_user_embeddings = model.user_embedding.get_weights()[0]
    old_user_biases = model.user_bias.get_weights()[0]
    old_book_embeddings = model.book_embedding.get_weights()[0]
    old_book_biases = model.book_bias.get_weights()[0]

    # Check for new users and books
    user2user_encoded, book2book_encoded = create_mappings(merged_df)

    new_num_users = len(user2user_encoded)
    new_num_books = len(book2book_encoded)

    # Initialize a new model with the updated number of users and books
    new_model = RecommenderNet(new_num_users, new_num_books, embedding_size)

    # Call the model once with some dummy data to create the variables
    dummy_input = np.array([[0, 0]])
    new_model(dummy_input)

    # Initialize new embeddings and biases with old values
    new_user_embeddings = np.zeros((new_num_users, embedding_size))
    new_user_biases = np.zeros((new_num_users, 1))
    new_book_embeddings = np.zeros((new_num_books, embedding_size))
    new_book_biases = np.zeros((new_num_books, 1))

    new_user_embeddings[:old_num_users] = old_user_embeddings
    new_user_biases[:old_num_users] = old_user_biases
    new_book_embeddings[:old_num_books] = old_book_embeddings
    new_book_biases[:old_num_books] = old_book_biases

    new_model.user_embedding.set_weights([new_user_embeddings])
    new_model.user_bias.set_weights([new_user_biases])
    new_model.book_embedding.set_weights([new_book_embeddings])
    new_model.book_bias.set_weights([new_book_biases])

    # Compile the new model
    new_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mse', 'accuracy']
    )

    return new_model

# Main routine to train or update the model
def main():
    merged_df = prepare_data()

    # Load previous mappings if they exist
    try:
        user2user_encoded, book2book_encoded = load_mappings()
    except FileNotFoundError:
        user2user_encoded, book2book_encoded = create_mappings(merged_df)
    
    user2user_encoded, book2book_encoded = update_mappings(merged_df, user2user_encoded, book2book_encoded)

    # Save updated mappings
    save_mappings(user2user_encoded, book2book_encoded)

    x_train, x_val, y_train, y_val, num_users, num_books = preprocess_data(merged_df, user2user_encoded, book2book_encoded)

    try:
        print('Load the previously saved model')
        model = tf.keras.models.load_model('Colab_User', custom_objects={'RecommenderNet': RecommenderNet})
        
        # Update the model for new users or books
        model = update_model_for_new_data(model, merged_df, model.num_users, model.num_books)
    except:
        print('Train a new model')
        model = train_model(x_train, y_train, x_val, y_val, num_users, num_books)

    # Perform one training step with the new data
    model.train_on_batch(x_train, y_train)

    # Save the retrained model
    model.save('Colab_User', save_format='tf')

if __name__ == '__main__':
    main()
