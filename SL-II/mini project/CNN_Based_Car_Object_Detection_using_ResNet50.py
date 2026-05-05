import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
# ResNet50 requires a standard input size of 224x224
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20

# ==========================================
# 2. MODEL ARCHITECTURE (Transfer Learning)
# ==========================================
def build_object_detector(optimizer_name='adam'):
    """
    Builds a ResNet50 model with two output heads:
    1. Classification (Is it a car?)
    2. Bounding Box Regression (Where is the car?)
    """
    # Load pre-trained ResNet50 without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)

    # Freeze the base model layers to prevent destroying pre-trained weights
    base_model.trainable = False

    # Extract features from the base model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Head 1: Classification (1 = Car, 0 = Background)
    class_output = layers.Dense(1, activation='sigmoid', name='class_output')(x)

    # Head 2: Bounding Box Regression (x_min, y_min, x_max, y_max)
    # Linear activation is used because coordinates are continuous numbers
    bbox_output = layers.Dense(4, activation='linear', name='bbox_output')(x)

    # Combine into a single multi-output model
    model = models.Model(inputs=base_model.input, outputs=[class_output, bbox_output])

    # Configure the selected optimizer
    if optimizer_name == 'adam':
        opt = optimizers.Adam(learning_rate=1e-4)
    elif optimizer_name == 'sgd':
        opt = optimizers.SGD(learning_rate=1e-4, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=1e-4)

    # Compile with distinct loss functions for each head
    model.compile(
        optimizer=opt,
        loss={
            'class_output': 'binary_crossentropy', # For classification
            'bbox_output': 'mean_squared_error'    # MSE for coordinate regression
        },
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mse'
        }
    )
    return model

# ==========================================
# 3. DATA PREPARATION (Mock Data for Structure)
# ==========================================
# NOTE: Replace this section with your actual Kaggle Dataset parsing code.
# You will need to load images, resize them, and parse the XML/CSV bounding boxes.
print("Generating mock dataset for testing pipeline...")
num_samples = 150
X_train = np.random.rand(num_samples, 224, 224, 3) # 150 Random Images
y_class_train = np.random.randint(0, 2, size=(num_samples, 1)) # 150 Labels
y_bbox_train = np.random.rand(num_samples, 4) # 150 Bounding Boxes [x, y, w, h]

# ==========================================
# 4. TRAINING & COMPARATIVE ANALYSIS
# ==========================================
def train_and_compare():
    optimizers_to_test = ['adam', 'sgd', 'rmsprop']
    history_dict = {}

    for opt in optimizers_to_test:
        print(f"\n=====================================")
        print(f" Training Model with Optimizer: {opt.upper()}")
        print(f"=====================================")

        # Build a fresh model for each optimizer
        model = build_object_detector(optimizer_name=opt)

        # Train the model
        history = model.fit(
            X_train,
            {'class_output': y_class_train, 'bbox_output': y_bbox_train},
            validation_split=0.2, # Use 20% of data for validation
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        # Store the history for plotting later
        history_dict[opt] = history.history

    return history_dict

# ==========================================
# 5. VISUALIZATION METRICS
# ==========================================
def plot_results(history_dict):
    plt.figure(figsize=(15, 6))

    # Subplot 1: Classification Accuracy Comparison
    plt.subplot(1, 2, 1)
    for opt, hist in history_dict.items():
        plt.plot(hist['val_class_output_accuracy'], label=f'{opt.upper()} (Validation)')
    plt.title('Classification Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Bounding Box Regression Loss (MSE)
    plt.subplot(1, 2, 2)
    for opt, hist in history_dict.items():
        plt.plot(hist['val_bbox_output_loss'], label=f'{opt.upper()   } (Validation)')
    plt.title('Bounding Box Regression Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION COMMANDS
# ==========================================
# Run the comparison
results = train_and_compare()

# Plot the comparative graphs for your report
plot_results(results)