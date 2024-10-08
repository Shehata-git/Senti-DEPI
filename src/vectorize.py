import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
# from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from scipy.sparse import save_npz

final_df = pd.read_csv('csv file here')

def process_and_save_vectorized_data_huggingface(df, text_column, label_column, output_file, model_name='distilbert-base-uncased', batch_size=32):
    """
    Process text data using Hugging Face's transformer models and save the vectorized form and labels to CSV file.

    Parameters:
    - df: DataFrame containing the text data.
    - text_column: The name of the column containing text data.
    - label_column: The name of the column containing labels (e.g., sentiment).
    - output_file: The base name of the output files for vectorized data and labels.
    - model_name: The pre-trained transformer model name from Hugging Face (default: distilbert-base-uncased).
    - batch_size: Batch size for processing data (default: 32).
    """

    # Step 1: Fill NaN values in the text_column with an empty string
    df[text_column] = df[text_column].fillna('').astype(str)  # Ensure all data is of string type

    # Load the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Function to get embeddings
    def get_embeddings(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of the last hidden states as embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    # Process in batches to avoid memory overload
    all_embeddings = []
    for i in tqdm(range(0, len(df), batch_size), desc="Vectorizing Text"):
        batch_texts = df[text_column][i:i+batch_size].tolist()
        embeddings = get_embeddings(batch_texts)
        all_embeddings.append(embeddings)

        # Concatenate all embeddings
    all_embeddings = torch.tensor(all_embeddings).reshape(-1, 768).numpy()

    # Create DataFrame from embeddings and combine with labels
    embeddings_df = pd.DataFrame(all_embeddings)
    labels_df = df[[label_column]].reset_index(drop=True)

    # Combine features and labels
    final_df = pd.concat([labels_df, embeddings_df], axis=1)

    # Save to CSV
    final_df.to_csv(output_file + '.csv', index=False)
    print(f"Data saved to {output_file}.csv")

process_and_save_vectorized_data_huggingface(final_df, 'Processed_Review', 'Sentiment', 'vectorized_data_hf', batch_size=32)
