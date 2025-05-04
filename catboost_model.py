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

def extract_collaboration_info(text):
    """Extract collaboration information from text"""
    if pd.isna(text):
        return {
            'has_collab': 0,
            'collab_brand': 'none',
            'collab_type': 'none'
        }
    
    text = clean_text(text)
    
    # Common collaboration indicators
    collab_patterns = [
        r'collaboration',
        r'collab',
        r'partnership',
    ]
    
    # High-value collaborators
    premium_collaborators = {
        'travis scott': 3,
        'kanye west': 3,
        'kanye': 3,
        'jordan': 2,
        'pharrell williams': 2,
        'fragment': 3,
        'off-white': 3,
        'supreme': 3,
        'fear of god': 2,
        'bape': 2,
        'stussy': 2,
        'undefeated': 2,
        'kith': 2
    }
    
    # Check for collaboration
    has_collab = 0
    collab_brand = 'none'
    collab_type = 'none'
    
    for pattern in collab_patterns:
        match = re.search(pattern, text)
        if match:
            has_collab = 1
            if len(match.groups()) > 0:
                collab_brand = match.group(1).strip().lower()
                
                # Determine collaboration type/value
                for brand, value in premium_collaborators.items():
                    if brand in collab_brand:
                        collab_type = f'tier_{value}'
                        break
                if collab_type == 'none':
                    collab_type = 'tier_1'
                
            break
    
    return {
        'has_collab': has_collab,
        'collab_brand': collab_brand,
        'collab_type': collab_type
    }

def create_success_score(row):
    """Create a composite success score for sneakers"""
    
    # Base score from price premium (40% weight)
    price_premium = row['annual_avg_price']/row['retail_price'] if not pd.isna(row['annual_price_premium']) and row['retail_price'] > 0 else 0
    premium_score = min(max(price_premium, -1), 3) / 3  # Normalize to [0,1]
    
    # Sales velocity score (30% weight)
    sales_score = 0
    if not pd.isna(row['annual_sales_count']):
        sales_score = min(row['annual_sales_count'] / 1000, 1)  # Cap at 10k sales
    
    # Volatility penalty (15% weight)
    volatility_score = 0
    if not pd.isna(row['annual_volatility']):
        volatility_score = 1 - min(row['annual_volatility'], 1)  # Lower volatility is better
    
    # Profit potential (15% weight)
    profit_score = 0
    if not pd.isna(row['annual_avg_price']) and not pd.isna(row['retail_price']) and row['retail_price'] > 0:
        profit_potential = (row['annual_avg_price'] - row['retail_price']) / row['retail_price']
        profit_score = min(max(profit_potential, 0), 3) / 3  # Normalize to [0,1]
    
    # Combine scores with weights
    final_score = (
        0.15 * premium_score +
        0.15 * sales_score +
        0.3 * volatility_score +
        0.4 * profit_score
    )
    
    return final_score * 100  # Scale to 0-100

def prepare_features(df):
    """Prepare features for the model"""
    print("Preparing features...")
    
    # Clean text and extract retail prices
    df['clean_description'] = df['description'].apply(clean_text)
    df['retail_price'] = df['clean_description'].apply(extract_retail_price)
    
    # Remove rows where retail price couldn't be extracted
    df = df.dropna(subset=['retail_price'])
    
    # Extract collaboration information
    collab_info = df['clean_description'].apply(extract_collaboration_info)
    df['has_collaboration'] = collab_info.apply(lambda x: x['has_collab'])
    df['collab_brand'] = collab_info.apply(lambda x: x['collab_brand'])
    df['collab_type'] = collab_info.apply(lambda x: x['collab_type'])
    
    # Process dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['days_since_release'] = (datetime.now() - df['release_date']).dt.days
    df['release_month'] = df['release_date'].dt.month
    df['release_quarter'] = df['release_date'].dt.quarter
    
    # Create price features
    df['log_retail_price'] = np.log1p(df['retail_price'])
    
    # Brand statistics
    brand_stats = df.groupby('brand').agg({
        'annual_avg_price': ['mean', 'std'],
        'annual_price_premium': ['mean', 'std']
    }).reset_index()
    
    brand_stats.columns = ['brand', 'brand_price_mean', 'brand_price_std', 
                          'brand_ptr_mean', 'brand_ptr_std']
    
    df = df.merge(brand_stats, on='brand', how='left')
    
    # Calculate z-scores
    df['brand_price_zscore'] = (df['annual_avg_price'] - df['brand_price_mean']) / df['brand_price_std']
    df['brand_ptr_zscore'] = (df['annual_price_premium'] - df['brand_ptr_mean']) / df['brand_ptr_std']
    
    # Model encoding
    df['model'] = df['model'].fillna('Unknown')
    model_counts = df['model'].value_counts()
    df['model_processed'] = df['model'].apply(lambda x: x if x in model_counts[model_counts >= 1].index else 'Other')
    
    # Create success score
    df['success_score'] = df.apply(create_success_score, axis=1)
    
    # Select features for the model
    feature_columns = [
        # Basic features
        'retail_price', 'days_since_release', 'release_month', 'release_quarter',
        'log_retail_price', 'has_collaboration',
        
        # Brand and model features
        'brand_price_mean', 'brand_price_std', 'brand_ptr_mean', 'brand_ptr_std',
        'brand_price_zscore', 'brand_ptr_zscore',
        
        # Categorical features (for CatBoost)
        'brand', 'model_processed', 'collab_type'
    ]
    
    # Ensure all numeric columns have valid values
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
    
    return df, feature_columns

def train_catboost_model(df, feature_columns, target_column='success_score'):
    """Train and evaluate the CatBoost model"""
    print("Training CatBoost model...")
    
    # Prepare data
    X = df[feature_columns]
    y = df[target_column]
    pd.concat([X, y], axis=1).to_csv('processed_data.csv', index=False)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define categorical features
    cat_features = [
        'brand', 'model_processed', 'collab_type'
    ]
    
    # Initialize and train model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100
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
    
    return model, feature_importance

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('third_version_of_stockx_data_with_kinda_many_brands_without_dropping_nan_description.csv')
    
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