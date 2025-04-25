import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModel
import pickle

# Configure retry strategy for robust downloads
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)

# Create session with headers and retries
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=retry_strategy))
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Referer': 'https://www.stockx.com/'
})

# Image processing
def process_image(url):
    """Robust image processing with error handling"""
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        return img
    
    except Exception as e:
        print(f"Image download failed: {url} - {str(e)}")
        return None

# Define image preprocessing pipeline
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature Extraction Models
class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Image CNN model (ResNet50)
        self.cnn_model = models.resnet50(pretrained=True)
        self.cnn_model = nn.Sequential(*list(self.cnn_model.children())[:-1])  # Remove the last FC layer
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # Text model (BERT)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_model.to(self.device)
        self.text_model.eval()
    
    def extract_image_features(self, url):
        """Extract image features using ResNet50"""
        img = process_image(url)
        if img is None:
            return torch.zeros(2048, device=self.device)
        
        try:
            # Apply transforms and predict
            img_tensor = image_transforms(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.cnn_model(img_tensor)
                
            return features.squeeze().flatten()
        
        except Exception as e:
            print(f"Feature extraction failed: {url} - {str(e)}")
            return torch.zeros(2048, device=self.device)
    
    def extract_text_features(self, text):
        """Extract text features using BERT"""
        if pd.isna(text) or text == "":
            return torch.zeros(768, device=self.device)
        
        try:
            # Truncate text if too long
            text = str(text)[:512]
            
            # Tokenize and extract features
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # Use the CLS token embedding as the text representation
                text_features = outputs.last_hidden_state[:, 0, :]
                
            return text_features.squeeze()
        
        except Exception as e:
            print(f"Text feature extraction failed: {str(e)}")
            return torch.zeros(768, device=self.device)

def preprocess_data(df):
    """Enhanced data preprocessing pipeline for tabular features"""
    # Extract retail price
    price_pattern = r'(?:retail|price|at)\s*\$?(\d{2,3})(?:\.\d{2})?'
    df['retail_price'] = df['description'].str.extract(price_pattern, flags=re.IGNORECASE)[0].astype(float)
    
    # Handle release dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.fillna(-1).astype(int)
    df['release_month'] = df['release_date'].dt.month.fillna(-1).astype(int)
    df['release_dayofweek'] = df['release_date'].dt.dayofweek.fillna(-1).astype(int)
    
    # Encode categorical features
    df = pd.get_dummies(df, columns=['brand', 'product_category'], 
                       prefix=['brand', 'category'], dummy_na=True)
    
    return df

# Combined Model 
class SneakerPricePredictor(nn.Module):
    def __init__(self, tabular_dim, text_dim=768, img_dim=2048):
        super(SneakerPricePredictor, self).__init__()
        
        # CNN feature processing
        self.cnn_fc = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Text feature processing
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Tabular feature processing
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined layers
        self.combined_fc = nn.Sequential(
            nn.Linear(512 + 512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, img_features, text_features, tabular_features):
        img_out = self.cnn_fc(img_features)
        text_out = self.text_fc(text_features)
        tab_out = self.tabular_fc(tabular_features)
        
        # Concatenate all features
        combined = torch.cat((img_out, text_out, tab_out), dim=1)
        
        # Final prediction
        return self.combined_fc(combined)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('third_version_of_stockx_data_with_kinda_many_brands_without_dropping_nan_description.csv')
    
    # Process tabular features
    df = preprocess_data(df)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(device=device)
    
    # Extract image features
    print("Extracting image features...")
    image_features_list = []
    for url in tqdm(df['thumb_url'], desc="Processing images"):
        image_features = extractor.extract_image_features(url)
        # Move to CPU for storage
        image_features_list.append(image_features.cpu().numpy())
    
    # Extract text features from description
    print("Extracting text features...")
    text_features_list = []
    for text in tqdm(df['description'], desc="Processing descriptions"):
        text_features = extractor.extract_text_features(text)
        # Move to CPU for storage
        text_features_list.append(text_features.cpu().numpy())
    
    # Convert to arrays
    image_features_array = np.array(image_features_list)
    text_features_array = np.array(text_features_list)
    
    # Save features to disk
    print("Saving extracted features...")
    with open('image_features.pkl', 'wb') as f:
        pickle.dump(image_features_array, f)
    
    with open('text_features.pkl', 'wb') as f:
        pickle.dump(text_features_array, f)
    
    # Create and save tabular features
    tabular_cols = [col for col in df.columns if col.startswith(('brand_', 'category_', 'retail_price', 'release_'))]
    tabular_features = df[tabular_cols]
    
    # Normalize numerical features
    for col in ['retail_price', 'release_year', 'release_month', 'release_dayofweek']:
        if col in tabular_features.columns:
            tabular_features[col] = (tabular_features[col] - tabular_features[col].mean()) / tabular_features[col].std()
    
    tabular_features.to_csv('tabular_features.csv', index=False)
    
    print("Features extraction completed!")
    print(f"Image features shape: {image_features_array.shape}")
    print(f"Text features shape: {text_features_array.shape}")
    print(f"Tabular features shape: {tabular_features.shape}")

if __name__ == "__main__":
    main() 