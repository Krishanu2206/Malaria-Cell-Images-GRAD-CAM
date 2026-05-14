# Malaria Cell Classification with Grad-CAM

A deep learning project for classifying malaria-infected blood cells using a custom CNN with Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanations.

## 📊 Project Overview

This project implements a binary image classification model to automatically detect parasitized vs. uninfected blood cells from the **Kaggle Malaria Cell Images dataset**. The model not only makes predictions but also provides visual explanations through Grad-CAM heatmaps, showing which regions of the cell influenced the classification decision.

**Dataset:** [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

**Grad-CAM official paper :** [Paper] (https://arxiv.org/pdf/1610.02391)

## 🎯 Model Performance

### Training Results (10 Epochs)

| Metric                  | Train                  | Test   |
| ----------------------- | ---------------------- | ------ |
| **Final Accuracy**      | 95.84%                 | 95.95% |
| **Final Loss**          | 0.1304                 | 0.1296 |
| **Total Training Time** | ~2 hours 16 minutes    | -      |
| **Hardware**            | CUDA (GPU-accelerated) | -      |

### Epoch-by-Epoch Progress

```
Epoch 1  | train_loss: 0.2707 | train_acc: 90.21% | test_loss: 0.1579 | test_acc: 95.31%
Epoch 2  | train_loss: 0.1641 | train_acc: 95.16% | test_loss: 0.1440 | test_acc: 95.81%
Epoch 3  | train_loss: 0.1567 | train_acc: 95.22% | test_loss: 0.1633 | test_acc: 95.43%
Epoch 4  | train_loss: 0.1474 | train_acc: 95.44% | test_loss: 0.1605 | test_acc: 95.79%
Epoch 5  | train_loss: 0.1454 | train_acc: 95.58% | test_loss: 0.1433 | test_acc: 94.95%
Epoch 6  | train_loss: 0.1426 | train_acc: 95.52% | test_loss: 0.1243 | test_acc: 95.98%
Epoch 7  | train_loss: 0.1386 | train_acc: 95.84% | test_loss: 0.1353 | test_acc: 96.11%
Epoch 8  | train_loss: 0.1358 | train_acc: 95.55% | test_loss: 0.1338 | test_acc: 95.96%
Epoch 9  | train_loss: 0.1351 | train_acc: 95.71% | test_loss: 0.1475 | test_acc: 95.99%
Epoch 10 | train_loss: 0.1304 | train_acc: 95.84% | test_loss: 0.1296 | test_acc: 95.95%
```

## 🏗️ Model Architecture

The model is a custom CNN with 6 convolutional blocks followed by fully connected layers:

```
MalariaClassifier(
  ConvBlock(3 → 64)     → [MaxPool]
  ConvBlock(64 → 64)    → [MaxPool]
  ConvBlock(64 → 128)   → [MaxPool]
  ConvBlock(128 → 256)  → [MaxPool]
  ConvBlock(256 → 512)  → [MaxPool]
  ConvBlock(512 → 64)   → [MaxPool]
  Flatten
  Linear(256 → 256)     → [ReLU, Dropout(0.2)]
  Linear(256 → 2)       → [Output: Parasitized/Uninfected]
)
```

Each `ConvBlock` contains:

- Dropout (0.2)
- Conv2d (kernel=3×3, padding=1)
- ReLU activation
- BatchNorm2d
- MaxPool2d (2×2)

## 📁 Project Structure

```
Malaria-Cell-Images-GRAD-CAM/
├── notebook.ipynb              # Jupyter notebook with full training pipeline
├── streamlit_app.py            # Interactive web app for inference
├── models/
│   └── malaria_grad_cam.pth    # Trained model weights (saved after training)
├── data/
│   ├── cell_images/            # Original dataset (train/test split)
│   ├── train/                  # Training images split by class
│   └── test/                   # Test images split by class
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision
pip install streamlit
pip install scikit-learn
pip install opencv-python
pip install Pillow numpy matplotlib tqdm
```

### Step 1: Train the Model

Open and run `notebook.ipynb`:

1. Install dependencies (first cell)
2. Load and preprocess the dataset
3. Create and train the model (10 epochs)
4. Save the trained weights to `models/malaria_grad_cam.pth`

The model training will output epoch-by-epoch metrics and save the checkpoint automatically.

### Step 2: Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will:

- Open in your browser (usually `http://localhost:8501`)
- Show controls in the left sidebar
- Load the pre-trained model with GPU support (optional)
- Accept uploaded cell images
- Display predictions and Grad-CAM visualizations

## 🎨 Features

### Notebook (`notebook.ipynb`)

- ✅ **Data Loading & Preprocessing** - Image transforms with normalization and augmentation
- ✅ **Model Architecture** - Custom CNN with residual concepts
- ✅ **Training Loop** - Full training pipeline with metrics tracking
- ✅ **Evaluation** - Per-epoch train/test metrics
- ✅ **Model Saving** - Checkpoint with metadata (class names, indices)
- ✅ **Grad-CAM Visualization** - Heatmap generation and overlay
- ✅ **Single Image Prediction** - `predict_and_show_cam()` helper function

### Streamlit App (`streamlit_app.py`)

- ✅ **User-Friendly Interface** - Upload images easily
- ✅ **Batch Processing** - Process multiple images at once
- ✅ **Real-time Predictions** - Instant classification
- ✅ **Grad-CAM Overlay** - Side-by-side original and heatmap display
- ✅ **GPU Support** - Optional CUDA acceleration
- ✅ **Error Handling** - Graceful error messages
- ✅ **Session Caching** - Efficient model loading

## 📊 Data Details

### Dataset Statistics

- **Classes:** 2 (Parasitized, Uninfected)
- **Input Size:** 128×128 RGB images
- **Train/Test Split:** Randomized 80/20 split
- **Total Images:** ~27,558 cell images from Kaggle

### Image Preprocessing

**Training Transform:**

```python
Resize(128×128)
CenterCrop(128×128)
ColorJitter(brightness±5%, contrast±5%)
RandomAffine(rotation±5°, translation±10%)
Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

**Validation/Test Transform:**

```python
Resize(128×128)
CenterCrop(128×128)
Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

## 🔬 How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps showing which image regions were most important for the model's prediction:

1. **Forward Pass** - Extract activation maps from the last convolutional layer
2. **Backward Pass** - Compute gradients of the predicted class w.r.t. activation maps
3. **Weight Computation** - Calculate average gradient per feature map (importance weights)
4. **Heatmap Generation** - Weighted combination of activation maps
5. **Overlay** - Blend heatmap with original image for visualization

### Interpretation

- **Red/Hot regions** - High activation (model focused here)
- **Blue/Cool regions** - Low activation (model ignored)
- Helps validate that the model learns relevant features (cell structure, parasites)

## 💻 Usage Examples

### In the Streamlit App

1. Click "Load Model Weights" in the sidebar (default path: `models/malaria_grad_cam.pth`)
2. Upload 1+ cell images (PNG, JPG, JPEG)
3. View predictions and Grad-CAM heatmaps instantly

### In the Notebook

```python
# Single image prediction from test dataset
predict_and_show_cam(dataset_index=0)

# Prediction from file path
predict_and_show_cam(image_path='path/to/cell_image.png')

# Generate Grad-CAM for multiple images
N = 20
test_dataloader = DataLoader(test_dataset, batch_size=N, shuffle=False)
images, labels = next(iter(test_dataloader))

for i in range(N):
    heatmap, pred = im2gradCAM(images[i:i+1].to(device))
    # ... visualization code
```

## 🔧 Configuration

### Model Hyperparameters

- **Learning Rate:** 0.01 (Adam optimizer)
- **Batch Size:** 32
- **Epochs:** 10
- **Loss Function:** CrossEntropyLoss
- **Input Size:** 128×128
- **Dropout Rate:** 0.2

### Device Configuration

- Automatic GPU detection (CUDA if available, else CPU)
- Streamlit app allows manual GPU toggle
- Notebook auto-detects device

## 📦 Model Checkpoint Format

The saved checkpoint contains:

```python
{
    'model_state_dict': {...},          # Model weights
    'class_names': ['Parasitized', 'Uninfected'],
    'class_to_idx': {'Parasitized': 0, 'Uninfected': 1},
    'num_classes': 2,
    'model_config': {
        'architecture': 'MalariaClassifier',
        'input_size': 128,
        'num_epochs_trained': 10
    }
}
```

## 🐛 Troubleshooting

### Model Not Found Error

- Ensure you've run the notebook training cell
- Verify `models/malaria_grad_cam.pth` exists
- Check the file path in streamlit app settings

### Out of Memory (OOM)

- Reduce batch size in notebook (default: 32)
- Use CPU mode in streamlit app (uncheck GPU toggle)
- Reduce image resolution (currently 128×128)

### Slow Inference

- Enable GPU acceleration (toggle in streamlit sidebar)
- Reduce number of images processed at once
- Ensure CUDA drivers are up-to-date

### Image Upload Issues

- Ensure images are RGB (converted automatically)
- File size: ~50-100KB recommended
- Format: PNG, JPG, JPEG

## 📚 References

- **Dataset:** https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- **Grad-CAM Paper:** https://arxiv.org/abs/1610.02055
- **PyTorch:** https://pytorch.org/
- **Streamlit:** https://streamlit.io/

## 🎓 Learning Resources

- Understanding CNN architectures
- Gradient-based explanation methods
- Data augmentation strategies
- Model deployment with Streamlit

## 📝 License

This project is based on the Kaggle Malaria Cell Images dataset.

## ✨ Future Improvements

- [ ] Model ensemble for higher accuracy
- [ ] Attention mechanisms
- [ ] Multi-class extension (more parasite types)
- [ ] API endpoint (FastAPI/Flask)
- [ ] Docker containerization
- [ ] Model quantization for mobile deployment
- [ ] Active learning loop

---

**Last Updated:** May 2026  
**Author:** Modern Computer Vision with PyTorch - Ch. 6  
**Status:** ✅ Completed & Tested
