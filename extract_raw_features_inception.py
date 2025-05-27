import pandas as pd
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
from tqdm import tqdm
from PIL import Image
import pickle

def create_feature_extractor():
    """Create and return InceptionV3 feature extractor."""
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # Use the output of the last convolutional layer
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
    return model

def extract_features(model, img_path):
    """Extract features from image file."""
    try:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def main():
    # Create feature extractor
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor()
    
    # Get list of image files
    image_dir = 'product_images1'
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Extract features
    print("Extracting features from images...")
    features_dict = {}
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        url_key = os.path.splitext(img_file)[0]  # Use filename without extension as url_key
        features = extract_features(feature_extractor, img_path)
        if features is not None:
            features_dict[url_key] = features
    
    # Save raw features
    print("\nSaving raw features...")
    with open('raw_inception_features.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"Raw features saved for {len(features_dict)} images")
    print("Feature dimension:", len(next(iter(features_dict.values()))))

if __name__ == "__main__":
    main() 
