# 🚗 Driver Behavior Detection using CNN and Pre-trained Models

This project focuses on detecting driver behavior from images using Convolutional Neural Networks (CNNs). The dataset consists of images categorized into five classes: `other_activities`, `safe_driving`, `talking_phone`, `texting_phone`, and `turning`. The goal is to classify these behaviors accurately to enhance road safety by identifying potentially dangerous actions like using a phone while driving.

<img width="1203" height="886" alt="image" src="https://github.com/user-attachments/assets/40bc0f1a-1eda-49cf-86d8-3ea0030de7d9" />
<img width="389" height="389" alt="image" src="https://github.com/user-attachments/assets/eea0782a-a547-4f81-a901-85137f2bc20b" />


**The notebook implements and compares three different CNN architectures:**
1. 🛠️ A custom CNN model
2. 🏛️ A VGG-like model
3. 🔗 A ResNet34-inspired model

Each model is trained on the dataset, evaluated, and compared based on accuracy and other metrics.

## 📚 Imports and Libraries Used

The project relies on several Python libraries for data handling, image processing, and deep learning:

- 🔢 **numpy**: For numerical operations and array manipulations.
- 📊 **pandas**: For creating and managing DataFrames to organize image paths and labels.
- 📁 **os**: For interacting with the file system to load image paths.
- 🖼️ **PIL.Image**: For opening and converting images.
- 📈 **matplotlib.pyplot**: For visualizing images and plotting results like confusion matrices and accuracy comparisons.
- 🖼️ **skimage.io.imread**: For reading images in various formats.
- ✂️ **sklearn.model_selection.train_test_split**: For splitting the dataset into training, validation, and test sets while maintaining class balance.
- 📊 **sklearn.metrics**: For evaluating model performance, including classification reports and confusion matrices.
- 🤖 **tensorflow.keras**: The core library for building and training neural networks:
  - **Layers**: Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization for constructing the model architectures.
  - **Models**: Sequential and Model for defining the CNN structures.
  - **Preprocessing**: ImageDataGenerator for data augmentation and preprocessing.
  - **Optimizers**: Adamax and Adam for optimizing the models during training.
  - **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint for controlling the training process.

These libraries provide a comprehensive toolkit for image classification tasks, from data preparation to model evaluation.


### 📁 Loading Image Paths

The dataset is stored in a directory structure where each class has its own folder containing images. The base path is set to `/kaggle/input/revitsone-5class/Revitsone-5classes`, and the classes are:
- 🤸 `other_activities`
- 🚗 `safe_driving`
- 📞 `talking_phone`
- 📱 `texting_phone`
- ↩️ `turning`

A function `load_images(folder)` is defined to collect all image file paths (with `.png` or `.jpg` extensions) from a given folder. This function iterates through the files in the specified folder and returns a list of full paths to the images.

The code then creates a dictionary `images_dict` where each key is a category, and the value is a list of image paths for that category. It prints the number of images in each category to verify the data loading.

### 🧹 Removing Unwanted Images

To ensure data quality, certain images that are corrupted, mislabeled, or unsuitable are identified and removed. Lists of unwanted image filenames are defined for specific categories:
- `unwanted_others`: Images to remove from `other_activities`.
- `unwanted_turning`: Images to remove from `turning`.

A helper function `remove_unwanted(images_list, unwanted_names)` filters out images whose filenames match the unwanted list. This function is applied to the respective image lists, resulting in cleaned datasets for each category.

### 📋 Creating a Labeled DataFrame

All image paths and their corresponding labels are combined into a pandas DataFrame for easier manipulation. A dictionary `all_classes` maps each class name to its list of image paths. Using list comprehension, a list of tuples `(image_path, label)` is created for all images across all classes. This list is then converted into a DataFrame with columns `image_path` and `label`.

The DataFrame provides a structured way to handle the dataset, allowing for easy splitting and feeding into the model training pipeline.

### ✂️ Splitting Data into Train, Validation, and Test Sets

The dataset is split into three parts using stratified sampling to maintain class distribution:
1. **Train-Test Split**: 80% for training/validation, 20% for testing.
2. **Train-Validation Split**: From the training set, 80% for training, 20% for validation.

This results in:
- `train_df`: Used for training the models.
- `valid_df`: Used for hyperparameter tuning and monitoring overfitting.
- `test_df`: Used for final evaluation.

The sizes of each set are printed to confirm the splits.

### 👀 Visualizing Random Images

To inspect the dataset, four random images from the `turning` class are selected and displayed in a 2x2 grid using matplotlib. This helps verify that images are loading correctly and provides a visual understanding of the data.

### 🔄 Image Augmentation and Preprocessing

Images are preprocessed using `ImageDataGenerator` from Keras:
- **Rescaling**: Pixel values are normalized to [0, 1] by dividing by 255.
- **Generators**: Separate generators are created for training, validation, and test sets.
  - Training generator: Includes shuffling for randomness.
  - Validation and test generators: No shuffling to ensure consistent evaluation.

The generators use `flow_from_dataframe` to create batches of images and labels directly from the DataFrame. Images are resized to 240x240 pixels, and the class mode is set to "categorical" for multi-class classification. Batch sizes are 64 for training and 32 for validation/test.

## 🏗️ Model Architecture and Building

Three different CNN models are implemented and compared.

### 🛠️ Custom CNN Model

A custom CNN is built using the Sequential API:
- **Architecture**:
  - Four convolutional blocks, each with two Conv2D layers (increasing filters: 32, 64, 128, 256), BatchNormalization, MaxPooling, and Dropout.
  - Flatten layer to convert 2D features to 1D.
  - Dense layer with 512 neurons, BatchNormalization, Dropout.
  - Output layer with 5 neurons and softmax activation.
- **Purpose**: Designed to extract hierarchical features while preventing overfitting with regularization techniques.

The model summary is printed to show the layer details and parameter counts.

### 🏛️ VGG-like CNN Model

Inspired by the VGG architecture:
- **Architecture**:
  - Four blocks of convolutional layers with increasing filters (64, 128, 256, 512).
  - Each block has multiple Conv2D layers followed by BatchNormalization and MaxPooling.
  - Flatten, two Dense layers with 4096 neurons and Dropout, and the output layer.
- **Purpose**: Follows the VGG design for deep feature extraction with Dropout to control overfitting.

### 🔗 ResNet34-like CNN Model

Using the Functional API to implement a ResNet34-inspired architecture:
- **Architecture**:
  - Input layer for 240x240 RGB images.
  - Stem: Conv2D, BatchNormalization, ReLU, MaxPooling.
  - Residual blocks with skip connections: 3 blocks of 64 filters, 4 of 128, 6 of 256, 3 of 512.
  - GlobalAveragePooling and Dense output.
- **Purpose**: Utilizes residual connections to ease gradient flow in deep networks, improving training stability.

## 🚀 Training Process

### 🏃‍♂️ Custom CNN Training
- Compiled with Adam optimizer, categorical cross-entropy loss, and accuracy metric.
- Trained with EarlyStopping (patience=3) to prevent overfitting.
- Fits on `train_gen` with validation on `valid_gen` for 20 epochs.

### 🏃‍♀️ VGG-like Training
- Compiled with Adamax optimizer (learning rate=0.001).
- EarlyStopping with patience=5.
- Trained similarly for 20 epochs.

### 🏃 ResNet34 Training
- Compiled with Adam optimizer (learning rate=1e-4).
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.3, patience=3), ModelCheckpoint to save the best model.
- Trained for 25 epochs.

After training, each model is evaluated on train and test sets, printing loss and accuracy.

## 📈 Evaluation and Results

### 💾 Model Saving and Loading
All trained models are saved in `.keras` format for later use.

### 🔮 Prediction on Sample Image
A sample image is loaded, preprocessed, and predictions are made using all three models. The predicted class is printed for each model.

### 📊 Detailed Evaluation
For each model:
- Predictions are generated on the test set.
- Accuracy is calculated.
- Classification report is printed, showing precision, recall, F1-score per class.
- Confusion matrix is visualized using matplotlib, showing true vs. predicted labels.

### 🏆 Model Comparison
A bar chart compares the accuracies of the three models. A summary table lists the final accuracies.

This comprehensive evaluation helps identify the best-performing model for the task.</content>

[View the original Kaggle notebook](https://www.kaggle.com/code/abdelrahmanmahmoud22/driver-behavior-detection-cnn)


## 👤 Author

**HOSEN ARAFAT**  

**Software Engineer, China**  

**GitHub:** https://github.com/arafathosense

**Researcher: Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, Image Processing**
