import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import pickle
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import requests
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def clean_text(text):
    if pd.isna(text) or text is None:
        return "unknown"
    soup = BeautifulSoup(str(text), 'html.parser')
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_retail_price(text):
    if pd.isna(text):
        return np.nan
    patterns = [
        r'\$\s*(\d{2,4}(?:\.\d{2})?)',
        r'retail.*?\$(\d{2,4}(?:\.\d{2})?)',
        r'priced at \$?(\d{2,4}(?:\.\d{2})?)'
    ]
    text = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                price = float(match.group(1))
                if 20 <= price <= 10000:
                    return price
            except:
                continue
    return np.nan

def prepare_initial_features(df):
    print("Preparing initial features...")
    df['description'] = df['description'].fillna('unknown')
    df['clean_description'] = df['description'].apply(clean_text)
    df['retail_price'] = df['clean_description'].apply(extract_retail_price)
    df = df.dropna(subset=['retail_price'])
    df['resell_price'] = df['annual_avg_price']
    df = df.dropna(subset=['resell_price'])
    return df

def download_images(df, output_dir='images', image_type='small'):
    """
    Download product images from CSV URLs
    
    Parameters:
    - csv_path: path to CSV file
    - output_dir: directory to save images
    - image_type: 'small' (300x214) or 'thumb' (140x100)
    """
    
    # Read CSV data
    df = df
    print(f"Loaded {len(df)} products from CSV")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure URL column and target dimensions
    url_column = 'small_image_url' if image_type == 'small' else 'thumb_url'
    
    # Download images with progress bar
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        try:
            url = row[url_column]
            url_key = row['url_key']
            filename = f"{output_dir}/{url_key}.jpg"
            
            # Skip existing files
            if os.path.exists(filename):
                skipped_count += 1
                continue
                
            # Download image
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Save image
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            success_count += 1
            
        except Exception as e:
            error_count += 1
            continue

    # Print summary        
    print(f"\nDownload complete:")
    print(f"Successfully downloaded: {success_count}")
    print(f"Skipped existing files: {skipped_count}")
    print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    df = pd.read_csv('12136_pairs_without_dropping_nan_descr.csv')
    df = prepare_initial_features(df)
    download_images(
        df=df,
        output_dir='product_images1',
        image_type='small'  # Change to 'thumb' for smaller images
    )
