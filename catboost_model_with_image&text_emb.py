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
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap.umap_ as umap
import os
import requests
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import io
import colorsys

# df = pd.read_csv('12136_pairs_without_dropping_nan_descr.csv')
# df['title'] = df['title'].str.lower()
# df['name'] = df['name'].str.lower()
# df.to_csv('12136_pairs_lowercase.csv', index=False)

def clean_text(text):
    """Clean text by removing HTML tags and normalizing whitespace"""
    if pd.isna(text) or text is None:
        return "unknown"
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
        'off-white': 3,
        'yeezy': 3,
        'clot': 3,
        'fragment': 3,
        'supreme': 3,
        'travis scott': 3,
        'drake': 3,
        'louis vuitton': 3,
        
        'rick owens': 2,
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
    if pd.isna(title) or title is None:
        return 'men'
    title = str(title).lower()
    
    # Define size categories and their keywords
    size_categories = {
        'gs': ['gs', 'grade school', 'big kids'],
        'women': ['women', 'womens', 'wmns', 'wmn'],
        'ps': ['ps', 'preschool', 'little', 'kids'],
        'td': ['td', 'toddler'],
        'infants': ['infant', 'infants'],
        'men': []  # Default category
    }
    
    # Check for each category
    for category, keywords in size_categories.items():
        for keyword in keywords:
            if keyword in title:
                return category
    
    return 'men'  # Default to men if no category found

def vectorize_titles(df, n_topics=8, max_features=100):
    """
    Векторизует заголовки с помощью TF-IDF + NMF для интерпретируемой тематической модели
    
    Параметры:
    - df: исходный датафрейм
    - n_topics: количество тем для извлечения
    - max_features: максимальное количество слов для TF-IDF
    
    Возвращает:
    - df с добавленными колонками тем
    - объект tfidf
    - объект nmf
    - список слов (признаков) из TF-IDF
    """
    print("Векторизация заголовков (title)...")
    
    # Подготовка текста
    titles = df['title'].fillna('unknown').astype(str).str.lower()
    
    # TF-IDF векторизация
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # униграммы и биграммы
        min_df=5,  # минимальная частота документов
    )
    tfidf_matrix = tfidf.fit_transform(titles)
    feature_names = tfidf.get_feature_names_out()
    
    # NMF для тематического моделирования
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf_matrix = nmf.fit_transform(tfidf_matrix)
    
    # Интерпретация тем (топ-7 слов на тему)
    print("\nИнтерпретация тем из заголовков:")
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[:-8:-1]  # 7 слов с наибольшим весом
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Тема {topic_idx+1}: {', '.join(top_words)}")
    
    # Добавить темы как признаки
    for i in range(n_topics):
        df[f'title_topic_{i+1}'] = nmf_matrix[:, i]
    
    return df, tfidf, nmf, feature_names

def reduce_image_features_with_pca(df, n_components=10):
    """
    Сокращает размерность визуальных признаков с помощью PCA с предварительным масштабированием
    n_components=5 выбрано для минимизации переобучения и сохранения >80% дисперсии
    """
    img_feature_cols = [col for col in df.columns if col.startswith('inception_')]
    if not img_feature_cols:
        print("Визуальные признаки не найдены!")
        return df, None, []
    img_features = df[img_feature_cols].fillna(0)
    scaler = StandardScaler()
    img_features_scaled = scaler.fit_transform(img_features)
    pca = PCA(n_components=n_components, random_state=42)
    img_features_pca = pca.fit_transform(img_features_scaled)
    for i in range(n_components):
        df[f'img_pca_{i+1}'] = img_features_pca[:, i]
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print(f"\nАнализ визуальных компонент PCA:")
    print(f"Объясненная дисперсия по компонентам: {explained_variance[:5].round(3)}")
    print(f"Суммарная объясненная дисперсия: {cumulative_variance[-1]:.2%}")
    print("\nИнтерпретация визуальных компонент (топ-5 признаков по весу):")
    for i, component in enumerate(pca.components_[:5]):
        top_indices = np.argsort(np.abs(component))[::-1][:5]
        print(f"\nКомпонента {i+1} (вес в дисперсии: {explained_variance[i]:.2%}):")
        for idx in top_indices:
            print(f"  {img_feature_cols[idx]}: weight={component[idx]:.4f}")
    return df, pca, img_features_scaled, img_feature_cols

def reduce_image_features_with_umap(df, img_features_scaled, n_components=8):
    """
    Применяет UMAP для нелинейного сокращения размерности визуальных признаков
    n_components=3 выбрано для минимизации переобучения
    """
    print("\nПрименяем UMAP для нелинейного сокращения размерности визуальных признаков...")
    reducer = umap.UMAP(n_components=n_components, random_state=42, min_dist=0.1, n_neighbors=15)
    umap_embedding = reducer.fit_transform(img_features_scaled)
    for i in range(n_components):
        df[f'img_umap_{i+1}'] = umap_embedding[:, i]
    print(f"UMAP-признаки добавлены: {n_components} компонент")
    return df, reducer

def extract_color_features_from_inception(df, img_features, img_feature_cols):
    """
    Извлекает цветовые характеристики на основе InceptionV3 признаков
    
    InceptionV3 имеет много фильтров, чувствительных к цвету. 
    Мы используем кластеризацию для определения доминирующих цветовых характеристик.
    """
    print("\nИзвлечение цветовых характеристик из InceptionV3 признаков...")
    
    # Выделим каналы, наиболее чувствительные к цвету
    # Эти индексы часто соответствуют цветовым фильтрам в InceptionV3
    # (на основе анализа архитектуры сети)
    color_sensitive_indices = [
        # RGB-чувствительные индексы
        [i for i in range(100, 200)],   # красный канал
        [i for i in range(200, 300)],   # зеленый канал
        [i for i in range(300, 400)],   # синий канал
        # Более высокие слои для текстуры
        [i for i in range(1000, 1100)], # текстурные признаки
    ]
    
    # Вычислим средние значения по каждой группе цветочувствительных фильтров
    for i, indices in enumerate(color_sensitive_indices[:3]):  # Первые 3 группы - RGB
        channel_features = np.mean([img_features[:, idx-1] for idx in indices if idx-1 < len(img_feature_cols)], axis=0)
        color_name = ['red', 'green', 'blue'][i]
        df[f'color_{color_name}_intensity'] = channel_features
    
    # Извлекаем доминирующие цвета с помощью кластеризации
    # Выбираем подмножество цветовых признаков
    color_features = df[['color_red_intensity', 'color_green_intensity', 'color_blue_intensity']].values
    
    # Нормализуем значения RGB в диапазон [0, 1]
    color_min = np.min(color_features, axis=0)
    color_max = np.max(color_features, axis=0)
    color_features_norm = (color_features - color_min) / (color_max - color_min + 1e-10)
    
    # Применяем кластеризацию для определения доминирующих цветов
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(color_features_norm)
    
    # Добавляем метки кластеров в датафрейм
    df['color_cluster'] = cluster_labels
    
    # Добавляем HSV характеристики (оттенок, насыщенность, яркость)
    hsv_features = np.array([colorsys.rgb_to_hsv(r, g, b) 
                            for r, g, b in color_features_norm])
    
    df['color_hue'] = hsv_features[:, 0]        # оттенок
    df['color_saturation'] = hsv_features[:, 1]  # насыщенность
    df['color_brightness'] = hsv_features[:, 2]  # яркость
    
    # Вычисляем контрастность (стандартное отклонение по RGB)
    df['color_contrast'] = np.std(color_features_norm, axis=1)
    
    print(f"Добавлены цветовые характеристики: RGB интенсивность, HSV компоненты, кластеры цветов, контрастность")
    
    return df

def prepare_features(df):
    """Prepare features for the model"""
    print("Preparing features...")

    # Fill missing text fields early
    df['description'] = df['description'].fillna('unknown')
    df['title'] = df['title'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')
    df['model'] = df['model'].fillna('Unknown')
    df['name'] = df['name'].str.lower()
    # Clean text and extract retail prices
    df['clean_description'] = df['description'].apply(clean_text)
    df['retail_price'] = df['clean_description'].apply(extract_retail_price)
    df = df.dropna(subset=['retail_price'])

    # Extract collaboration information
    collab_info = df.apply(extract_collaboration_info, axis=1)
    df['has_collaboration'] = collab_info.apply(lambda x: x['has_collab'])
    df['collab_count'] = collab_info.apply(lambda x: x['collab_count'])
    df['collab_tier'] = collab_info.apply(lambda x: x['highest_tier'])
    df['collab_brands'] = collab_info.apply(lambda x: ','.join(x['collab_brands']) if x['collab_brands'] else 'none')

    # Process dates robustly
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)
    df['release_quarter'] = df['release_date'].dt.quarter.fillna(0).astype(int)

    return df

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
def prepare_initial_features(df):
    print("Preparing initial features...")
    # Fill missing text fields early
    df['description'] = df['description'].fillna('unknown')
    df['title'] = df['title'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')
    df['model'] = df['model'].fillna('Unknown')

    # Clean text and extract retail prices
    df['clean_description'] = df['description'].apply(clean_text)
    df['retail_price'] = df['clean_description'].apply(extract_retail_price)
    df = df.dropna(subset=['retail_price'])

    # Extract collaboration info
    collab_info = df.apply(extract_collaboration_info, axis=1)
    df['has_collaboration'] = collab_info.apply(lambda x: x['has_collab'])
    df['collab_count'] = collab_info.apply(lambda x: x['collab_count'])
    df['collab_tier'] = collab_info.apply(lambda x: x['highest_tier'])
    df['collab_brands'] = collab_info.apply(lambda x: ','.join(x['collab_brands']) if x['collab_brands'] else 'none')
    df['brand'] = df['brand'].apply(lambda x: x.lower())

    # Process model names
    df['model'] = df['model'].fillna('Unknown')
    df['model_processed'] = df.apply(clean_model_name, axis=1)

    # Extract size category
    df['size_category'] = df['title'].apply(extract_size_category)

    # Create resell_price
    df['resell_price'] = df['annual_avg_price']
    df = df.dropna(subset=['resell_price'])

    # Process dates robustly
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)
    df['release_quarter'] = df['release_date'].dt.quarter.fillna(0).astype(int)

    return df

def train_catboost_model(X_train, X_test, y_train, y_test, feature_columns):
    print("Training CatBoost model...")
    cat_features = [
        'brand', 
        'model_processed', 
        'collab_brands', 
        'size_category',
        'color_cluster',
    ]
    feature_weights = {}
    for col in feature_columns:
        if col.startswith('img_pca_') or col.startswith('img_umap_'):
            feature_weights[col] = 2
        elif col.startswith('color_'):
            feature_weights[col] = 2
    # Увеличиваем регуляризацию
    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=8,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=100,
        feature_weights=feature_weights if feature_weights else None,
        early_stopping_rounds=500
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
    )
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    return model, feature_importance

def main():
    print("Loading data...")
    df = pd.read_csv('12136_pairs_without_dropping_nan_descr.csv')
    df = prepare_initial_features(df)
    df, tfidf, nmf, title_features = vectorize_titles(df, n_topics=8, max_features=100)
    
    # Load raw Inception features
    print("\nLoading raw Inception features...")
    with open('raw_inception_features.pkl', 'rb') as f:
        features_dict = pickle.load(f)
        
        # Convert dictionary to DataFrame
        features_df = pd.DataFrame.from_dict(features_dict, orient='index')
        features_df.index.name = 'url_key'
        
        # Add inception_ prefix to column names
        features_df.columns = [f'inception_{i}' for i in range(features_df.shape[1])]
        
        # Merge features with main dataframe
        df = df.merge(features_df, on='url_key', how='left')
        
        # PCA до 10 компонент
        df, pca, img_features_scaled, img_feature_cols = reduce_image_features_with_pca(df, n_components=10)
        # UMAP до 8 компонент
        df, umap_reducer = reduce_image_features_with_umap(df, img_features_scaled, n_components=8)
        df = extract_color_features_from_inception(df, img_features_scaled, img_feature_cols)
        
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    categorical_cols = ['brand', 'model_processed', 'collab_brands', 'size_category', 'clean_description']
    if 'color_cluster' in df.columns:
        categorical_cols.append('color_cluster')
    for col in categorical_cols:
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
    numeric_cols = [
        'brand_retail_mean', 'brand_retail_median', 'brand_resell_mean', 'brand_resell_median',
        'model_retail_mean', 'model_retail_median', 'model_resell_mean', 'model_resell_median',
        'model_count', 'release_month', 'release_quarter',
        'retail_to_brand_mean', 'model_resell_mean_to_brand_mean', 'retail_to_model_mean'
    ]
    for col in numeric_cols:
        train_median = train_df[col].median() if col in train_df else 0
        if col in train_df:
            train_df[col] = train_df[col].fillna(train_median)
        if col in test_df:
            test_df[col] = test_df[col].fillna(train_median)
    color_cols = [col for col in df.columns if col.startswith('color_')]
    for col in color_cols:
        if col in train_df.columns:
            train_median = train_df[col].median()
            train_df[col] = train_df[col].fillna(train_median)
            test_df[col] = test_df[col].fillna(train_median)
    model_counts_train = train_df['model_processed'].value_counts()
    popular_models_train = model_counts_train[model_counts_train >= 1].index
    train_df['model_processed'] = train_df['model_processed'].apply(lambda x: x if x in popular_models_train else 'Other')
    test_df['model_processed'] = test_df['model_processed'].apply(lambda x: x if x in popular_models_train else 'Other')
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
    title_topic_cols = [f'title_topic_{i+1}' for i in range(8)]
    img_pca_cols = [f'img_pca_{i+1}' for i in range(10)] if pca is not None else []
    img_umap_cols = [f'img_umap_{i+1}' for i in range(8)] if umap_reducer is not None else []
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
    ] + title_topic_cols + img_pca_cols + img_umap_cols + color_cols
    print(f"\nИспользуемые признаки ({len(feature_columns)}):")
    print(f"- Табличные: {len(feature_columns) - len(title_topic_cols) - len(img_pca_cols) - len(img_umap_cols) - len(color_cols)}")
    print(f"- Текстовые (из заголовков): {len(title_topic_cols)}")
    print(f"- Визуальные PCA: {len(img_pca_cols)}")
    print(f"- Визуальные UMAP: {len(img_umap_cols)}")
    print(f"- Цветовые: {len(color_cols)}")
    X_train = train_df[feature_columns]
    y_train = train_df['resell_price']
    X_test = test_df[feature_columns]
    y_test = test_df['resell_price']
    model, feature_importance = train_catboost_model(X_train, X_test, y_train, y_test, feature_columns)
    print("\nТоп-20 важнейших признаков:")
    top_features = feature_importance.head(20)
    for idx, row in top_features.iterrows():
        feature_type = ""
        if row['feature'] in title_topic_cols:
            topic_idx = int(row['feature'].split('_')[-1]) - 1
            top_words = [title_features[i] for i in nmf.components_[topic_idx].argsort()[:-4:-1]]
            feature_type = f"[Текст] Тема: {', '.join(top_words)}"
        elif row['feature'] in img_pca_cols:
            feature_type = "[Изображение PCA]"
        elif row['feature'] in img_umap_cols:
            feature_type = "[Изображение UMAP]"
        elif row['feature'] in color_cols:
            feature_type = "[Цвет]"
        print(f"{row['feature']}: {row['importance']:.4f} {feature_type}")
    importance_by_type = {
        'Табличные': feature_importance[~feature_importance['feature'].isin(title_topic_cols + img_pca_cols + img_umap_cols + color_cols)]['importance'].sum(),
        'Текстовые': feature_importance[feature_importance['feature'].isin(title_topic_cols)]['importance'].sum(),
        'Визуальные PCA': feature_importance[feature_importance['feature'].isin(img_pca_cols)]['importance'].sum() if img_pca_cols else 0,
        'Визуальные UMAP': feature_importance[feature_importance['feature'].isin(img_umap_cols)]['importance'].sum() if img_umap_cols else 0,
        'Цветовые': feature_importance[feature_importance['feature'].isin(color_cols)]['importance'].sum() if color_cols else 0
    }
    print("\nСуммарная важность по типам признаков:")
    for type_name, importance in importance_by_type.items():
        print(f"{type_name}: {importance:.4f} ({importance/feature_importance['importance'].sum():.1%})")
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Тип признаков': importance_by_type.keys(),
        'Важность': importance_by_type.values()
    })
    sns.barplot(x='Важность', y='Тип признаков', data=importance_df)
    plt.title('Важность признаков по типам')
    plt.tight_layout()
    plt.savefig('feature_importance_by_type.png')
    model.save_model('sneaker_success_model.cbm')
    feature_importance.to_csv('feature_importance.csv', index=False)
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test
    full_df = pd.concat([train_df, test_df], axis=0)
    full_df.to_csv('full_dataset.csv', index=False)
    with open('title_topics.txt', 'w') as f:
        f.write("Темы из заголовков кроссовок:\n")
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [title_features[i] for i in top_words_idx]
            f.write(f"Тема {topic_idx+1}: {', '.join(top_words)}\n")
    print("Модель обучена и сохранена. Результаты анализа записаны в файлы.")

if __name__ == "__main__":
    main()
