# Sneaker Resell Price Predictor

A deep learning system that predicts resell prices of sneakers using a multimodal approach, combining image features, text descriptions, and tabular data.

## Architecture

The prediction model uses a combined architecture that processes:

1. **Image Data** - Uses a CNN (ResNet50) to extract visual features from sneaker images
2. **Text Data** - Uses a transformer model (DistilBERT) to extract features from text descriptions
3. **Tabular Data** - Processes structured data like brand, release date, retail price, etc.

These features are concatenated and passed through multiple linear layers to predict the resell price.

## Project Structure

- `feature_extraction_pytorch.py` - Extracts features from images, text, and tabular data
- `train_model.py` - Trains the price prediction model using the extracted features
- `requirements.txt` - Lists all required Python dependencies

## Data Processing Pipeline

1. Images are processed through a ResNet50 model without the classification head to extract 2048-dimensional features
2. Text descriptions are processed through DistilBERT to extract 768-dimensional features
3. Tabular features are extracted and normalized
4. All features are combined and used to train a neural network for price prediction

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)

### Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```

### Usage

1. **Feature Extraction**:
```bash
python feature_extraction_pytorch.py
```
This will:
- Process the StockX dataset
- Extract image features using ResNet50
- Extract text features using DistilBERT
- Process tabular features
- Save all features to disk

2. **Model Training**:
```bash
python train_model.py
```
This will:
- Load the extracted features
- Train the price prediction model
- Evaluate the model on a test set
- Save the trained model and performance metrics

## Results

After training, you'll find:
- `sneaker_price_model.pt` - The trained PyTorch model
- `training_loss.png` - A plot of training and validation loss
- `prediction_results.png` - A scatter plot of predicted vs actual prices
- `model_metrics.txt` - Performance metrics (MAE, RMSE, R²)

## Dataset

The model is trained on StockX data containing information about sneakers, including:
- Images
- Text descriptions
- Brand information
- Release dates
- Retail prices
- Market prices (lowest ask, highest bid)

## Further Improvements

- Experiment with different CNN architectures for image processing
- Fine-tune the text embedding model for the sneaker domain
- Add more feature engineering for the tabular data
- Implement data augmentation for images
- Optimize hyperparameters using a systematic search 