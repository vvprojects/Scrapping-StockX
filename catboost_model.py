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

def extract_collaboration_info(row):
    """Extract collaboration information from text"""
    if pd.isna(row['description']) and pd.isna(row['title']):
        return {
            'has_collab': 0,
            'collab_brand': 'none',
            'collab_type': 'none'
        }
    
    # Combine title and description for better collaboration detection
    text = f"{str(row['title'])} {str(row['description'])}"
    text = clean_text(text)
    
    # Only use specified collaboration patterns
    collab_patterns = [
        r'collaboration',
        r'collab',
        r'partnership',
        r' x ',  # Common separator in titles    
    ]
    
    # Collaborator tiers based on cultural impact
    premium_collaborators = {
        'michael jordan': 3,
        'jordan': 3,
        'virgil abloh': 3,
        'off-white': 3,
        'kanye west': 3,
        'yeezy': 3,
        'fragment': 3,
        'supreme': 3,
        'travis scott': 3,
        'travis': 3,
        'cactus jack': 3,
        'drake': 3,
        'ovo': 3,
        'fear of god': 3,
        'fog': 3,
        'sacai': 3,
        'union': 3,
        'a ma maniere': 3,
        'aime leon dore': 3,
        'alexander wang': 3,
        'ambush': 3,
        'atmos': 3,
        'bape': 3,
        'billionaire boys club': 3,
        'bbc': 3,
        'cdg': 3,
        'comme des garcons': 3,
        'dior': 3,
        'fragment': 3,
        'kaws': 3,
        'kenzo': 3,
        'mastermind': 3,
        'mastermind japan': 3,
        'medicom': 3,
        'neighborhood': 3,
        'nbhd': 3,
        'nike sb': 3,
        'parra': 3,
        'patta': 3,
        'stussy': 3,
        'undefeated': 3,
        'wtaps': 3,
        
        'allen iverson': 2,
        'kobe bryant': 2,
        'lebron james': 2,
        'pharrell williams': 2,
        'tinker hatfield': 2,
        'andre agassi': 2,
        'gucci': 2,
        'louis vuitton': 2,
        'lv': 2,
        'prada': 2,
        'balenciaga': 2,
        'adidas yeezy': 2,
        
        'sheryl swoopes': 1,
        'ken griffey jr': 1
    }
    
    # Check for collaboration
    has_collab = 0
    collab_brand = 'none'
    collab_type = 'none'
    
    # First check for explicit collaboration mentions
    for pattern in collab_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # If we find a collaboration pattern, check for specific collaborators
            for brand, tier in premium_collaborators.items():
                if brand in text.lower():
                    has_collab = 1
                    collab_brand = brand
                    collab_type = f'tier_{tier}'
                    break
            if has_collab:
                break
    
    # If no collaboration found but we see a premium collaborator, it might be a signature series
    if not has_collab:
        for brand, tier in premium_collaborators.items():
            if brand in text.lower():
                # Check if it's likely a signature series (e.g., "Jordan 1", "Kobe 6")
                if re.search(rf'{brand}\s+\d+', text, re.IGNORECASE):
                    has_collab = 1
                    collab_brand = brand
                    collab_type = f'tier_{tier}'
                    break
    
    return {
        'has_collab': has_collab,
        'collab_brand': collab_brand,
        'collab_type': collab_type
    }

def extract_size_category(title):
    """Extract size category from title"""
    if pd.isna(title):
        return 'men'
    
    title = str(title).lower()
    
    # Define size categories and their keywords
    size_categories = {
        'GS': ['gs', 'grade school', 'big kids'],
        'Women': ['women', 'womens', 'wmns', 'wmn'],
        'PS': ['ps', 'preschool', 'little kids'],
        'TD': ['td', 'toddler'],
        'men': []  # Default category
    }
    
    # Check for each category
    for category, keywords in size_categories.items():
        for keyword in keywords:
            if keyword in title:
                return category
    
    return 'men'  # Default to men if no category found

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
    collab_info = df.apply(extract_collaboration_info, axis=1)
    df['has_collaboration'] = collab_info.apply(lambda x: x['has_collab'])
    df['collab_brand'] = collab_info.apply(lambda x: x['collab_brand'])
    df['collab_type'] = collab_info.apply(lambda x: x['collab_type'])
    
    # Extract size category
    df['size_category'] = df['title'].apply(extract_size_category)
    
    # Process dates
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    # df['days_since_release'] = ((datetime.now() - df['release_date']).dt.days).where(lambda x: x < 3650)  
    # df = df.dropna(subset=['days_since_release'])
    
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
    
    # Model encoding - extract clean model names
    df['model'] = df['model'].fillna('Unknown')
    
    def clean_model_name(row):
        model = str(row['model']).lower()
        brand = str(row['brand']).lower()
        
        # Remove brand name from model name
        if brand in model:
            model = model.replace(brand, '').strip()
        
        # Remove common prefixes/suffixes
        prefixes = ['nike', 'adidas', 'jordan', 'new balance', 'puma', 'reebok', 'converse']
        for prefix in prefixes:
            if model.startswith(prefix):
                model = model[len(prefix):].strip()
        
        # Clean up any remaining special characters and extra spaces
        model = re.sub(r'[^\w\s-]', '', model)
        model = re.sub(r'\s+', ' ', model).strip()
        
        return model if model else 'Unknown'
    
    # Apply model name cleaning
    df['model_processed'] = df.apply(clean_model_name, axis=1)
    
    # Group rare models
    model_counts = df['model_processed'].value_counts()
    df['model_processed'] = df['model_processed'].apply(lambda x: x if x in model_counts[model_counts >= 1].index else 'Other')
    
    # Create success score
    # df['success_score'] = df.apply(create_success_score, axis=1)
    df['resell_price'] = df['annual_avg_price'].where(lambda x: x > 0)  
    df = df.dropna(subset= ['resell_price'])
    
    # Select features for the model
    feature_columns = [
        # Basic features
        'retail_price', 
        # 'days_since_release',
        'release_month', 'release_quarter',
        'log_retail_price', 'has_collaboration',
        
        # Brand and model features
        'brand_price_mean', 'brand_price_std', 'brand_ptr_mean', 'brand_ptr_std',
        
        # Categorical features (for CatBoost)
        'brand', 'model_processed', 'collab_type', 'size_category',
        # 'clean_description'
    ]
    
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
        'brand', 'model_processed', 'collab_type', 'size_category'
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
        verbose=100
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
    print(f"MAE: {mae:.2f}$")
    print(f"RMSE: {rmse:.2f}$")
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