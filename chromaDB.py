from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch
from langchain_huggingface import HuggingFaceEmbeddings

# Load the CSV data
loader = CSVLoader(file_path="data_book_metadata.csv", encoding="utf-8")
data = loader.load()

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define the fine-tuning model
class FineTuningModel(nn.Module):
    def __init__(self, base_model):
        super(FineTuningModel, self).__init__()
        self.base_model = base_model
        self.dense = nn.Linear(384, 256)  # 384 is the hidden size of all-MiniLM-L6-v2

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        dense_output = self.dense(cls_token)
        return dense_output

# Initialize the fine-tuning model
fine_tuned_model = FineTuningModel(model)
fine_tuned_model.load_state_dict(torch.load('fine_tuned_embedding.pt'))
fine_tuned_model.eval()

# Create the embedding function using the fine-tuned model
# embedding_function = SentenceTransformerEmbeddings('fine_tuned_embedding')

model_kwargs = {'device':'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding_function = HuggingFaceEmbeddings(
    model_name= 'fine_tune_embedding.json',     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

# Specify the folder for the database
folder_db = "db"

# Load it into Chroma
db = Chroma.from_documents(data, fine_tuned_model, persist_directory=folder_db)

print("Data loaded into Chroma with fine-tuned embeddings.")
