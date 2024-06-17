import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras

# Load data
book_df = pd.read_csv('books_data_clean_with_id.csv')
rating_df = pd.read_csv('books_rating_clean_with_book_id.csv')
merged_df = pd.merge(rating_df, book_df, left_on='book_id', right_on='id')

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

EMBEDDING_SIZE = 64

@keras.utils.register_keras_serializable(package="RecommenderPackage")
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
        
        # The sigmoid activation forces the rating to be between 0 and 11
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

model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)

# Load the previously saved model
# model = tf.keras.models.load_model('Colab_User', custom_objects={'RecommenderNet': RecommenderNet(num_users, num_books, EMBEDDING_SIZE)})

# Compile the loaded model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mse', 'accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Continue training the model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32, 
    epochs=1,
    verbose=2,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

# Save the retrained model
model.save('Colab_User', save_format='tf')