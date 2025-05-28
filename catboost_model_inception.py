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

# --- Копируем функции подготовки данных и текстовых признаков из catboost_model.py ---
# (оставляем только нужные для pipeline)

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

def extract_collaboration_info(row):
    if pd.isna(row['description']) and pd.isna(row['title']):
        return {
            'has_collab': 0,
            'collab_brands': [],
            'collab_count': 0,
            'highest_tier': 0
        }
    text = f"{str(row['title'])} {str(row['description'])}"
    text = clean_text(text)
    collaborators = {
        'off-white': 3, 'yeezy': 3, 'clot': 3, 'fragment': 3, 'supreme': 3, 'travis scott': 3, 'drake': 3, 'louis vuitton': 3,
        'rick owens': 2, 'fear of god': 2, 'sacai': 2, 'union': 2, 'a ma maniere': 2, 'aime leon dore': 2, 'alexander wang': 2, 'ambush': 2, 'atmos': 2, 'bape': 2, 'billionaire boys club': 2, 'comme des garcons': 2, 'dior': 2, 'kaws': 2, 'kenzo': 2, 'mastermind': 2, 'medicom': 2, 'neighborhood': 2, 'nbhd': 2, 'parra': 2, 'patta': 2, 'stussy': 2, 'kobe bryant': 2, 'lebron james': 2, 'pharrell williams': 2, 'prada': 2, 'balenciaga': 2, 'gucci': 2, 'undefeated': 1
    }
    found_collabs = []
    highest_tier = 0
    title = str(row['title']).lower()
    for collab, tier in collaborators.items():
        if collab in title:
            if collab not in found_collabs:
                found_collabs.append(collab)
                highest_tier = max(highest_tier, tier)
    desc = str(row['description']).lower()
    for collab, tier in collaborators.items():
        if collab in desc:
            if collab not in found_collabs:
                found_collabs.append(collab)
                highest_tier = max(highest_tier, tier)
    return {
        'has_collab': 1 if found_collabs else 0,
        'collab_brands': found_collabs,
        'collab_count': len(found_collabs),
        'highest_tier': highest_tier
    }

def extract_size_category(title):
    if pd.isna(title) or title is None:
        return 'men'
    title = str(title).lower()
    size_categories = {
        'gs': ['gs', 'grade school', 'big kids'],
        'women': ['women', 'womens', 'wmns', 'wmn'],
        'ps': ['ps', 'preschool', 'little', 'kids'],
        'td': ['td', 'toddler'],
        'infants': ['infant', 'infants'],
        'men': []
    }
    for category, keywords in size_categories.items():
        for keyword in keywords:
            if keyword in title:
                return category
    return 'men'

def clean_model_name(row):
    model = str(row['model']).lower()
    brand = str(row['brand']).lower()
    model = model.replace(brand, '').strip()
    for collab in str(row['collab_brands']).lower().split(','):
        if collab != 'none':
            model = model.replace(collab, '').strip()
    model = re.sub(r'[^\w\s-]', '', model)
    model = re.sub(r'\s+', ' ', model).strip()
    return model if model else 'Unknown'

def prepare_initial_features(df):
    print("Preparing initial features...")
    df['description'] = df['description'].fillna('unknown')
    df['title'] = df['title'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')
    df['model'] = df['model'].fillna('Unknown')
    df['clean_description'] = df['description'].apply(clean_text)
    df['retail_price'] = df['clean_description'].apply(extract_retail_price)
    df = df.dropna(subset=['retail_price'])
    collab_info = df.apply(extract_collaboration_info, axis=1)
    df['has_collaboration'] = collab_info.apply(lambda x: x['has_collab'])
    df['collab_count'] = collab_info.apply(lambda x: x['collab_count'])
    df['collab_tier'] = collab_info.apply(lambda x: x['highest_tier'])
    df['collab_brands'] = collab_info.apply(lambda x: ','.join(x['collab_brands']) if x['collab_brands'] else 'none')
    df['brand'] = df['brand'].apply(lambda x: x.lower())
    df['model'] = df['model'].fillna('Unknown')
    df['model_processed'] = df.apply(clean_model_name, axis=1)
    df['size_category'] = df['title'].apply(extract_size_category)
    df['resell_price'] = df['annual_avg_price']
    df = df.dropna(subset=['resell_price'])
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)
    df['release_quarter'] = df['release_date'].dt.quarter.fillna(0).astype(int)
    return df

def vectorize_titles(df, n_topics=8, max_features=100):
    print("Векторизация заголовков (title)...")
    titles = df['title'].fillna('unknown').astype(str).str.lower()
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
    )
    tfidf_matrix = tfidf.fit_transform(titles)
    feature_names = tfidf.get_feature_names_out()
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf_matrix = nmf.fit_transform(tfidf_matrix)
    print("\nИнтерпретация тем из заголовков:")
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[:-8:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Тема {topic_idx+1}: {', '.join(top_words)}")
    for i in range(n_topics):
        df[f'title_topic_{i+1}'] = nmf_matrix[:, i]
    return df, tfidf, nmf, feature_names

# --- Основной pipeline ---
def main():
    print("Loading data...")
    df = pd.read_csv('12136_pairs_without_dropping_nan_descr.csv')
    df = prepare_initial_features(df)
    df, tfidf, nmf, title_features = vectorize_titles(df, n_topics=8, max_features=100)
    
    # Load experimental features
    experiment_dir = 'dimension_reduction_experiments'
    if not os.path.exists(experiment_dir):
        print("Error: dimension_reduction_experiments directory not found!")
        return
    
    # Get list of available experiments
    experiments = []
    for file in os.listdir(experiment_dir):
        if file.startswith('image_features_pca_'):
            experiment_name = file.replace('image_features_pca_', '').replace('.csv', '')
            experiments.append(experiment_name)
    
    if not experiments:
        print("Error: No experimental features found!")
        return
    
    print("\nAvailable experiments:")
    for exp in experiments:
        print(f"- {exp}")
    
    # Use the first experiment by default (you can modify this to use different experiments)
    experiment_name = experiments[0]
    print(f"\nUsing experiment: {experiment_name}")
    
    # Load PCA and UMAP features
    pca_features = pd.read_csv(f'{experiment_dir}/image_features_pca_{experiment_name}.csv', index_col='url_key')
    umap_features = pd.read_csv(f'{experiment_dir}/image_features_umap_{experiment_name}.csv', index_col='url_key')
    
    # Merge features
    df = df.merge(pca_features, on='url_key', how='left')
    df = df.merge(umap_features, on='url_key', how='left')
    
    print(f"\nVizual features added:")
    print(f"- PCA components: {len([c for c in df.columns if c.startswith('pca_')])}")
    print(f"- UMAP components: {len([c for c in df.columns if c.startswith('umap_')])}")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare categorical features
    categorical_cols = ['brand', 'model_processed', 'collab_brands', 'size_category', 'clean_description']
    for col in categorical_cols:
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
    
    # Prepare numeric features
    numeric_cols = [
        'brand_retail_mean', 'brand_retail_median', 'brand_resell_mean', 'brand_resell_median',
        'model_retail_mean', 'model_retail_median', 'model_resell_mean', 'model_resell_median',
        'model_count', 'release_month', 'release_quarter',
        'retail_to_brand_mean', 'model_resell_mean_to_brand_mean', 'retail_to_model_mean'
    ]
    
    # Add PCA and UMAP features to numeric columns
    pca_cols = [col for col in df.columns if col.startswith('pca_')]
    umap_cols = [col for col in df.columns if col.startswith('umap_')]
    numeric_cols.extend(pca_cols)
    numeric_cols.extend(umap_cols)
    
    for col in numeric_cols:
        train_median = train_df[col].median() if col in train_df else 0
        if col in train_df:
            train_df[col] = train_df[col].fillna(train_median)
        if col in test_df:
            test_df[col] = test_df[col].fillna(train_median)
    
    # Process model names
    model_counts_train = train_df['model_processed'].value_counts()
    popular_models_train = model_counts_train[model_counts_train >= 1].index
    train_df['model_processed'] = train_df['model_processed'].apply(lambda x: x if x in popular_models_train else 'Other')
    test_df['model_processed'] = test_df['model_processed'].apply(lambda x: x if x in popular_models_train else 'Other')
    
    # Calculate brand statistics
    brand_stats = train_df.groupby('brand').agg({
        'retail_price': ['mean', 'std', 'median'],
        'resell_price': ['mean', 'std', 'median']
    }).reset_index()
    brand_stats.columns = [
        'brand', 'brand_retail_mean', 'brand_retail_std', 'brand_retail_median',
        'brand_resell_mean', 'brand_resell_std', 'brand_resell_median'
    ]
    train_df = train_df.merge(brand_stats, on='brand', how='left')
    test_df = test_df.merge(brand_stats, on='brand', how='left')
    
    # Calculate model statistics
    model_stats = train_df.groupby('model_processed').agg({
        'retail_price': ['mean', 'std', 'median', 'count'],
        'resell_price': ['mean', 'std', 'median']
    }).reset_index()
    model_stats.columns = [
        'model_processed', 'model_retail_mean', 'model_retail_std', 
        'model_retail_median', 'model_count', 'model_resell_mean', 
        'model_resell_std', 'model_resell_median'
    ]
    train_df = train_df.merge(model_stats, on='model_processed', how='left')
    test_df = test_df.merge(model_stats, on='model_processed', how='left')
    
    # Prepare feature columns
    title_topic_cols = [f'title_topic_{i+1}' for i in range(8)]
    feature_columns = [
        'retail_price',
        'brand',
        'model_processed',
        'collab_brands',
        'size_category',
        'collab_tier',
        'collab_count', 
        'has_collaboration',
        'release_month',
        'release_quarter',
        'model_count',
    ] + title_topic_cols + pca_cols + umap_cols
    
    print(f"\nFeatures used ({len(feature_columns)}):")
    print(f"- Tabular: {len(feature_columns) - len(title_topic_cols) - len(pca_cols) - len(umap_cols)}")
    print(f"- Text (from titles): {len(title_topic_cols)}")
    print(f"- Visual PCA: {len(pca_cols)}")
    print(f"- Visual UMAP: {len(umap_cols)}")
    
    # Prepare data for training
    X_train = train_df[feature_columns]
    y_train = train_df['resell_price']
    X_test = test_df[feature_columns]
    y_test = test_df['resell_price']
    
    # Train model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=8,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=500
    )
    
    model.fit(
        X_train, y_train,
        cat_features=['brand', 'model_processed', 'collab_brands', 'size_category'],
        eval_set=(X_test, y_test),
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Analyze feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title(f'Feature Importance (Top 20) - {experiment_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{experiment_name}.png')
    
    # Analyze visual features importance
    print("\nTop-10 visual features:")
    visual_features = feature_importance[
        feature_importance['feature'].str.startswith(('pca_', 'umap_'))
    ].head(10)
    
    for idx, row in visual_features.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save results
    feature_importance.to_csv(f'feature_importance_{experiment_name}.csv', index=False)
    print(f"\nResults saved in feature_importance_{experiment_name}.csv")

if __name__ == "__main__":
    main() 
