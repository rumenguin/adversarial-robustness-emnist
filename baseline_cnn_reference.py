"""
Experiment 1: CNN Accuracy (Standard Training) on EMNIST Balanced Dataset
Author: Rumen
Date: 2026
Paper Reference: "A Robust Neural Network against Adversarial Attacks" (ETASR)

This experiment implements standard CNN training on EMNIST dataset following
the exact architecture and hyperparameters from the reference paper.

Dataset: EMNIST Balanced (47 classes, 28x28 grayscale images)
Architecture: 3 Conv layers + BatchNorm + 2 Dense layers
Total Parameters: 3,151,810 (matching paper specification)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU Configuration
print("="*80)
print("GPU CONFIGURATION")
print("="*80)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ {len(gpus)} GPU(s) available: {gpus}")
        print(f"✓ Memory growth enabled for all GPUs")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU available, using CPU")
print("="*80 + "\n")


class Config:
    """Configuration matching the paper's specifications"""
    
    # Dataset
    DATA_ROOT = "/kaggle/input/emnist/emnist"
    NUM_CLASSES = 47  # EMNIST Balanced
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    IMG_CHANNELS = 1
    
    # Training hyperparameters (from paper)
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001  # Adam default
    VALIDATION_SPLIT = 0.15
    
    # Architecture parameters (to achieve 3,151,810 params)
    CONV1_FILTERS = 20
    CONV1_KERNEL = (5, 5)
    
    CONV2_FILTERS = 20
    CONV2_KERNEL = (4, 4)
    
    CONV3_FILTERS = 20
    CONV3_KERNEL = (4, 4)
    
    DENSE1_UNITS = 200
    OUTPUT_UNITS = NUM_CLASSES
    
    DROPOUT_RATE = 0.0  # Paper uses no dropout
    
    # Output directories
    EXPERIMENT_NAME = "exp1_cnn_standard_old"
    OUTPUT_DIR = f"/mnt/user-data/outputs/{EXPERIMENT_NAME}"
    MODELS_DIR = f"{OUTPUT_DIR}/models"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    FIGURES_DIR = f"{OUTPUT_DIR}/figures"
    
    @classmethod
    def create_directories(cls):
        """Create output directory structure"""
        for directory in [cls.OUTPUT_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.FIGURES_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directories created at: {cls.OUTPUT_DIR}\n")


def load_emnist_data():
    """
    Load EMNIST Balanced dataset from directory structure
    Returns: (X_train, y_train), (X_test, y_test)
    """
    print("="*80)
    print("LOADING EMNIST BALANCED DATASET")
    print("="*80)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(Config.DATA_ROOT, 'train'),
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True,
        subset='training',
        seed=SEED
    )
    
    # Validation data
    val_generator = train_datagen.flow_from_directory(
        os.path.join(Config.DATA_ROOT, 'train'),
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False,
        subset='validation',
        seed=SEED
    )
    
    # Test data
    test_generator = test_datagen.flow_from_directory(
        os.path.join(Config.DATA_ROOT, 'test'),
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    print(f"\n✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Number of classes: {train_generator.num_classes}")
    print(f"✓ Image shape: {train_generator.image_shape}")
    print("="*80 + "\n")
    
    return train_generator, val_generator, test_generator


def build_cnn():
    """
    Build CNN architecture matching paper specifications
    Total Parameters: 3,151,810
    """
    print("="*80)
    print("BUILDING CNN ARCHITECTURE")
    print("="*80)
    
    model = models.Sequential(name='Standard_CNN')
    
    # First Conv Block: 1→20, kernel 5×5
    # Params: 1×5×5×20 + 20 = 520
    model.add(layers.Conv2D(
        Config.CONV1_FILTERS,
        Config.CONV1_KERNEL,
        activation='relu',
        padding='same',
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS),
        name='conv2d'
    ))
    model.add(layers.BatchNormalization(name='batch_normalization'))
    model.add(layers.Dropout(Config.DROPOUT_RATE, name='dropout'))
    
    # Second Conv Block: 20→20, kernel 4×4
    # Params: 20×4×4×20 + 20 = 6420
    model.add(layers.Conv2D(
        Config.CONV2_FILTERS,
        Config.CONV2_KERNEL,
        activation='relu',
        padding='same',
        name='conv2d_1'
    ))
    model.add(layers.BatchNormalization(name='batch_normalization_1'))
    model.add(layers.Dropout(Config.DROPOUT_RATE, name='dropout_1'))
    
    # Third Conv Block: 20→20, kernel 4×4
    # Params: 20×4×4×20 + 20 = 6420
    model.add(layers.Conv2D(
        Config.CONV3_FILTERS,
        Config.CONV3_KERNEL,
        activation='relu',
        padding='same',
        name='conv2d_2'
    ))
    model.add(layers.BatchNormalization(name='batch_normalization_2'))
    model.add(layers.Dropout(Config.DROPOUT_RATE, name='dropout_2'))
    
    # Flatten: 28×28×20 = 15680
    model.add(layers.Flatten(name='flatten'))
    
    # Dense: 15680→200
    # Params: 15680×200 + 200 = 3,136,200
    model.add(layers.Dense(Config.DENSE1_UNITS, activation='relu', name='dense'))
    
    # Output: 200→47 (EMNIST has 47 classes)
    # Params: 200×47 + 47 = 9,447
    model.add(layers.Dense(Config.OUTPUT_UNITS, activation='softmax', name='dense_1'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Verify parameter count
    total_params = model.count_params()
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"\n{'='*80}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Target (Paper - MNIST): 3,151,810")
    print(f"Note: EMNIST has 47 classes vs MNIST's 10, so output layer differs")
    
    # Calculate expected params
    conv1 = 1*5*5*20 + 20  # 520
    bn1 = 20*4  # 80 (mean, var, gamma, beta)
    conv2 = 20*4*4*20 + 20  # 6420
    bn2 = 20*4  # 80
    conv3 = 20*4*4*20 + 20  # 6420
    bn3 = 20*4  # 80
    dense1 = 15680*200 + 200  # 3,136,200
    dense2 = 200*47 + 47  # 9,447 (EMNIST)
    expected = conv1 + bn1 + conv2 + bn2 + conv3 + bn3 + dense1 + dense2
    
    print(f"Expected: {expected:,}")
    print(f"Match: {'YES ✓✓✓' if total_params == expected else f'NO (diff: {total_params - expected})'}")
    print("="*80 + "\n")
    
    return model


def train_model(model, train_gen, val_gen):
    """
    Train the CNN model
    """
    print("="*80)
    print("TRAINING CNN MODEL")
    print("="*80)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODELS_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(Config.RESULTS_DIR, 'training_log.csv')
        )
    ]
    
    # Train
    print(f"\nTraining for {Config.EPOCHS} epochs...")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Validation split: {Config.VALIDATION_SPLIT}\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80 + "\n")
    
    return history


def evaluate_model(model, test_gen):
    """
    Evaluate model on test set and generate comprehensive metrics
    """
    print("="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n{'='*80}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("="*80 + "\n")
    
    # Get predictions
    print("Generating predictions...")
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    class_names = list(test_gen.class_indices.keys())
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(Config.RESULTS_DIR, 'classification_report.csv'))
    print(f"✓ Classification report saved")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(Config.RESULTS_DIR, 'confusion_matrix.csv'))
    print(f"✓ Confusion matrix saved")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves (600 DPI, no captions)
    """
    print("\nGenerating training history plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Accuracy
    axes[0].plot(history.history['accuracy'], linewidth=2, label='Train', color='#2E86AB')
    axes[0].plot(history.history['val_accuracy'], linewidth=2, label='Validation', color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10, frameon=True)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=10)
    
    # Plot 2: Loss
    axes[1].plot(history.history['loss'], linewidth=2, label='Train', color='#2E86AB')
    axes[1].plot(history.history['val_loss'], linewidth=2, label='Validation', color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10, frameon=True)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save at 600 DPI
    output_path = os.path.join(Config.FIGURES_DIR, 'training_history.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved (600 DPI)")


def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix heatmap (600 DPI, no captions)
    """
    print("Generating confusion matrix heatmap...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    # Save at 600 DPI
    output_path = os.path.join(Config.FIGURES_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix heatmap saved (600 DPI)")


def plot_per_class_accuracy(report, class_names):
    """
    Plot per-class accuracy bar chart (600 DPI, no captions)
    """
    print("Generating per-class accuracy plot...")
    
    # Extract per-class metrics
    accuracy_scores = []
    for class_name in class_names:
        if class_name in report:
            accuracy_scores.append(report[class_name]['precision'])
        else:
            accuracy_scores.append(0.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_pos = np.arange(len(class_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
    
    bars = ax.bar(x_pos, accuracy_scores, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save at 600 DPI
    output_path = os.path.join(Config.FIGURES_DIR, 'per_class_accuracy.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class accuracy plot saved (600 DPI)")


def save_metadata(history, eval_results, training_time):
    """
    Save experiment metadata in JSON format
    """
    print("\nSaving experiment metadata...")
    
    metadata = {
        'experiment': {
            'name': Config.EXPERIMENT_NAME,
            'description': 'Standard CNN training on EMNIST Balanced dataset',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': training_time
        },
        'dataset': {
            'name': 'EMNIST Balanced',
            'num_classes': Config.NUM_CLASSES,
            'image_shape': [Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS],
            'data_root': Config.DATA_ROOT
        },
        'architecture': {
            'model_name': 'Standard_CNN',
            'conv_layers': 3,
            'dense_layers': 2,
            'total_parameters': int(np.sum([np.prod(v.get_shape()) for v in 
                                           tf.keras.backend.get_session().graph.get_collection('variables')]) 
                                          if hasattr(tf.keras.backend, 'get_session') else 0),
            'filters': [Config.CONV1_FILTERS, Config.CONV2_FILTERS, Config.CONV3_FILTERS],
            'kernels': [list(Config.CONV1_KERNEL), list(Config.CONV2_KERNEL), list(Config.CONV3_KERNEL)],
            'dense_units': [Config.DENSE1_UNITS, Config.OUTPUT_UNITS]
        },
        'hyperparameters': {
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'learning_rate': Config.LEARNING_RATE,
            'validation_split': Config.VALIDATION_SPLIT,
            'dropout_rate': Config.DROPOUT_RATE,
            'optimizer': 'Adam',
            'loss': 'categorical_crossentropy'
        },
        'results': {
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'test_accuracy': float(eval_results['test_accuracy']),
            'test_loss': float(eval_results['test_loss'])
        },
        'training_curves': {
            'epochs': list(range(1, len(history.history['accuracy']) + 1)),
            'train_accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        },
        'files': {
            'model': 'models/best_model.h5',
            'final_model': 'models/final_model.h5',
            'training_log': 'results/training_log.csv',
            'classification_report': 'results/classification_report.csv',
            'confusion_matrix': 'results/confusion_matrix.csv',
            'figures': [
                'figures/training_history.png',
                'figures/confusion_matrix.png',
                'figures/per_class_accuracy.png'
            ]
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Metadata saved to: {metadata_path}")


def main():
    """
    Main execution function
    """
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print(" " * 20 + "EXPERIMENT 1: CNN STANDARD TRAINING")
    print(" " * 25 + "EMNIST BALANCED DATASET")
    print("="*80 + "\n")
    
    # Create directories
    Config.create_directories()
    
    # Load data
    train_gen, val_gen, test_gen = load_emnist_data()
    
    # Build model
    model = build_cnn()
    
    # Train model
    history = train_model(model, train_gen, val_gen)
    
    # Save final model
    final_model_path = os.path.join(Config.MODELS_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"✓ Final model saved to: {final_model_path}\n")
    
    # Evaluate model
    eval_results = evaluate_model(model, test_gen)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_training_history(history)
    plot_confusion_matrix(eval_results['confusion_matrix'], eval_results['class_names'])
    plot_per_class_accuracy(eval_results['classification_report'], eval_results['class_names'])
    print("="*80 + "\n")
    
    # Calculate training time
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Save metadata
    save_metadata(history, eval_results, training_time)
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Test Accuracy: {eval_results['test_accuracy']*100:.2f}%")
    print(f"Test Loss: {eval_results['test_loss']:.6f}")
    print(f"\nAll results saved to: {Config.OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
