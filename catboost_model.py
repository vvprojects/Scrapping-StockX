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
from sklearn.decomposition import PCA

def clean_text(text):
    """Clean text by removing HTML tags and normalizing whitespace"""
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(str(text), 'html.parser')
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_retail_price(text):
    """Extract retail price from text"""
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

def extract_collaboration_info(row):
    """Extract collaboration information from text"""
    if pd.isna(row['description']) and pd.isna(row['title']):
        return {
            'has_collab': 0,
            'collab_brands': [],
            'collab_count': 0,
            'highest_tier': 0
        }
    
    # Combine title and description for better collaboration detection
    text = f"{str(row['title'])} {str(row['description'])}"
    text = clean_text(text)
    
    # List of collaborator patterns and their tiers
    collaborators = {
        'michael jordan': 3,
        'off-white': 3,
        'yeezy': 3,
        'clot': 3,
        'fragment': 3,
        'supreme': 3,
        'travis scott': 3,
        'drake': 3,
        'louis vuitton': 3,
        
        'ovo': 2,
        'fear of god': 2,
        'sacai': 2,
        'union': 2,
        'a ma maniere': 2,
        'aime leon dore': 2,
        'alexander wang': 2,
        'ambush': 2,
        'atmos': 2,
        'bape': 2,
        'billionaire boys club': 2,
        'comme des garcons': 2,
        'dior': 2,
        'kaws': 2,
        'kenzo': 2,
        'mastermind': 2,
        'medicom': 2,
        'neighborhood': 2,
        'nbhd': 2,
        'parra': 2,
        'patta': 2,
        'stussy': 2,
        'kobe bryant': 2,
        'lebron james': 2,
        'pharrell williams': 2,
        'prada': 2,
        'balenciaga': 2,
        'gucci': 2,
        'undefeated': 1
    }
    
    # Find all collaborations
    found_collabs = []
    highest_tier = 0
    
    # First check in title (higher priority)
    title = str(row['title']).lower()
    for collab, tier in collaborators.items():
        if collab in title:
            if collab not in found_collabs:  # Avoid duplicates
                found_collabs.append(collab)
                highest_tier = max(highest_tier, tier)
    
    # Then check in description
    desc = str(row['description']).lower()
    for collab, tier in collaborators.items():
        if collab in desc:
            if collab not in found_collabs:  # Avoid duplicates
                found_collabs.append(collab)
                highest_tier = max(highest_tier, tier)
    
    return {
        'has_collab': 1 if found_collabs else 0,
        'collab_brands': found_collabs,
        'collab_count': len(found_collabs),
        'highest_tier': highest_tier
    }

def extract_size_category(title):
    """Extract size category from title"""
    if pd.isna(title):
        return 'Men'
    
    title = str(title).lower()
    
    # Define size categories and their keywords
    size_categories = {
        'GS': ['gs', 'grade school', 'big kids'],
        'Women': ['women', 'womens', 'wmns', 'wmn'],
        'PS': ['ps', 'preschool', 'little kids'],
        'TD': ['td', 'toddler'],
        'Infants': ['infant', 'infants'],
        'Men': []  # Default category
    }
    
    # Check for each category
    for category, keywords in size_categories.items():
        for keyword in keywords:
            if keyword in title:
                return category
    
    return 'Men'  # Default to men if no category found

def create_success_score(row):
    """Create a composite success score for sneakers"""
    
    # Price premium score (15% weight)
    price_premium = row['annual_avg_price']/row['retail_price'] if not pd.isna(row['annual_price_premium']) and row['retail_price'] > 0 else 0
    premium_score = min(max(price_premium, -1), 3) / 3  # Normalize to [0,1]
    
    # Sales velocity score (15% weight)
    sales_score = 0
    if not pd.isna(row['annual_sales_count']):
        # Use log scale for sales to handle large variations
        log_sales = np.log1p(row['annual_sales_count'])
        max_log_sales = np.log1p(10000)  # Cap at 10k sales
        sales_score = min(log_sales / max_log_sales, 1)
    
    # Volatility penalty (30% weight)
    volatility_score = 0
    if not pd.isna(row['annual_volatility']):
        # Use exponential decay for volatility penalty
        volatility_score = np.exp(-row['annual_volatility'])
    
    # Profit potential (40% weight)
    profit_score = 0
    if not pd.isna(row['annual_avg_price']) and not pd.isna(row['retail_price']) and row['retail_price'] > 0 and row['annual_avg_price'] > 0:
        profit_potential = (row['annual_avg_price'] - row['retail_price']) / row['retail_price']
        # Use sigmoid function for smoother profit scoring
        profit_score = 1 / (1 + np.exp(-profit_potential))
    
    # Combine scores with weights by avg of PCA_weights,  Correlation_weights,  Regression_weights,  Ensemble_weights
    final_score = (
        0.357 * premium_score +
        0.123 * sales_score +
        0.088 * volatility_score +
        0.432 * profit_score
    )

    
    return final_score * 100  # Scale to 0-100

def prepare_features(df):
    """Prepare features for the model"""
    print("Preparing features...")
    
    # Load image features
    # try:
    #     with open('image_features_efficientnet.pkl', 'rb') as f:
    #         image_features = pickle.load(f)
    #         print(f"Loaded image features with shape: {image_features.shape}")
            
    #         # Apply PCA to reduce dimensionality
    #         pca = PCA(n_components=50)  # Reduce to 50 components
    #         scaler = StandardScaler()
    #         image_features_scaled = scaler.fit_transform(image_features)
    #         image_features_pca = pca.fit_transform(image_features_scaled)
    #         print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")
            
    #         # Create image feature columns
    #         for i in range(image_features_pca.shape[1]):
    #             df[f'img_feature_{i}'] = image_features_pca[:, i]
    # except FileNotFoundError:
    #     print("Warning: Image features not found. Run image_feature_extraction.py first.")
    #     image_features = None
    
    # Clean text and extract retail prices
    df['clean_description'] = df['description'].apply(clean_text)
    df['retail_price'] = df['clean_description'].apply(extract_retail_price)
    
    # Remove rows where retail price couldn't be extracted
    df = df.dropna(subset=['retail_price'])
    
    # Extract collaboration information
    collab_info = df.apply(extract_collaboration_info, axis=1)
    df['has_collaboration'] = collab_info.apply(lambda x: x['has_collab'])
    df['collab_count'] = collab_info.apply(lambda x: x['collab_count'])
    df['collab_tier'] = collab_info.apply(lambda x: x['highest_tier'])
    df['collab_brands'] = collab_info.apply(lambda x: ','.join(x['collab_brands']) if x['collab_brands'] else 'none')
    
    # Process dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month
    df['release_quarter'] = df['release_date'].dt.quarter
    
    # Create price features
    df['log_retail_price'] = np.log1p(df['retail_price'])
    
    # Brand statistics
    brand_stats = df.groupby('brand').agg({
        'retail_price': ['mean', 'std', 'median'],
        'annual_avg_price': ['mean', 'std', 'median']
    }).reset_index()
    
    brand_stats.columns = ['brand', 
                          'brand_retail_mean', 'brand_retail_std', 'brand_retail_median',
                          'brand_resell_mean', 'brand_resell_std', 'brand_resell_median']
    
    df = df.merge(brand_stats, on='brand', how='left')
    
    # Model encoding - extract clean model names
    df['model'] = df['model'].fillna('Unknown')
    
    def clean_model_name(row):
        model = str(row['model']).lower()
        brand = str(row['brand']).lower()
        
        # Remove brand name and collaborator names
        model = model.replace(brand, '').strip()
        for collab in str(row['collab_brands']).lower().split(','):
            if collab != 'none':
                model = model.replace(collab, '').strip()
        
        # Clean up any remaining special characters and extra spaces
        model = re.sub(r'[^\w\s-]', '', model)
        model = re.sub(r'\s+', ' ', model).strip()
        
        return model if model else 'Unknown'
    
    # Apply model name cleaning
    df['model_processed'] = df.apply(clean_model_name, axis=1)
    
    # Model statistics
    model_stats = df.groupby('model_processed').agg({
        'retail_price': ['mean', 'std', 'median', 'count'],
        'annual_avg_price': ['mean', 'std', 'median']
    }).reset_index()
    
    model_stats.columns = ['model_processed', 
                          'model_retail_mean', 'model_retail_std', 'model_retail_median', 'model_count',
                          'model_resell_mean', 'model_resell_std', 'model_resell_median']
    
    # Only keep models with sufficient data
    popular_models = model_stats[model_stats['model_count'] >= 5]['model_processed']
    df['model_processed'] = df['model_processed'].apply(lambda x: x if x in popular_models.values else 'Other')
    
    # Merge model stats
    df = df.merge(model_stats, on='model_processed', how='left')
    
    # Calculate price ratios
    df['retail_to_brand_mean'] = df['retail_price'] / df['brand_retail_mean']
    df['model_resell_mean_to_brand_mean'] = df['model_resell_mean'] / df['brand_resell_mean']
    df['retail_to_model_mean'] = df['retail_price'] / df['model_retail_mean']
    
    # Extract size category
    df['size_category'] = df['title'].apply(extract_size_category)
    
    # Create success score
    # df['success_score'] = df.apply(create_success_score, axis=1)
    df['resell_price'] = df['annual_avg_price'].where(lambda x: x > 0)  
    df = df.dropna(subset= ['resell_price'])
    
    # Select features for the model
    feature_columns = [
        # Basic features
        'retail_price', 
        'release_month', 'release_quarter',
        'log_retail_price',
        
        # Collaboration features
        'has_collaboration',
        'collab_count',
        'collab_tier',
        
        # Brand statistics
        'brand_retail_mean', 'brand_retail_std', 'brand_retail_median',
        'brand_resell_mean', 'brand_resell_std', 'brand_resell_median',
        
        # Model statistics
        'model_retail_mean', 'model_retail_std', 'model_retail_median',
        'model_resell_mean', 'model_resell_std', 'model_resell_median',
        'model_count',
        
        # Price ratios
        'retail_to_brand_mean', 'model_resell_mean_to_brand_mean',
        'retail_to_model_mean',
        
        # Categorical features
        'brand', 'model_processed', 'collab_brands', 'size_category'
    ]
    
    # Add image features if available
    # if image_features is not None:
    #     feature_columns.extend([f'img_feature_{i}' for i in range(50)])
    
    # Ensure all numeric columns have valid values
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
    
    return df, feature_columns

def train_catboost_model(df, feature_columns, target_column='resell_price'):
    """Train and evaluate the CatBoost model"""
    print("Training CatBoost model...")
    
    # Prepare data
    X = df[feature_columns]
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define categorical features
    cat_features = [
        'brand', 'model_processed', 'collab_brands', 'size_category'
    ]
    
    # Initialize and train model
    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='R2',
        random_seed=42,
        # text_features= ['clean_description'],
        verbose=20
    )
    
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=500
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    pd.concat([X, y], axis=1).to_csv('processed_data.csv', index=False)
    
    return model, feature_importance

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('fourth_version_of_stockx_data_with_kinda_many_brands_dropping_nan_description.csv')
    
    # Prepare features
    df, feature_columns = prepare_features(df)
    
    # Train model
    model, feature_importance = train_catboost_model(df, feature_columns)
    
    # Save model and feature importance
    model.save_model('sneaker_success_model.cbm')
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    print("\nModel and results saved to:")
    print("- sneaker_success_model.cbm")
    print("- feature_importance.csv")
    print("- feature_importance.png")

if __name__ == "__main__":
    main() 