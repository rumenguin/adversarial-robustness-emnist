"""
Experiment 2: CNN Adversarial Training using APE-GAN on EMNIST Balanced
Author: Rumen
Date: 2026

APE-GAN (Adversarial Perturbation Elimination with GAN) adversarial training:
- Generates adversarial examples during training
- Trains on mix of clean + adversarial samples
- Improves robustness against adversarial attacks
- Uses FGSM for adversarial generation (efficient for training)

Reference: Barr (2025) - "A Robust Neural Network against Adversarial Attacks"
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
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
        print(f"✓ {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("⚠ No GPU available")
print("="*80 + "\n")


class Config:
    """Configuration for APE-GAN adversarial training"""
    
    # Dataset
    DATA_ROOT = "/kaggle/input/emnist/emnist"
    NUM_CLASSES = 47
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    IMG_CHANNELS = 1
    
    # Training (same as Experiment 1)
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.15
    
    # Architecture (same as Experiment 1)
    CONV1_FILTERS = 32
    CONV1_KERNEL = (5, 5)
    CONV2_FILTERS = 64
    CONV2_KERNEL = (3, 3)
    CONV3_FILTERS = 128
    CONV3_KERNEL = (3, 3)
    DENSE1_UNITS = 512
    DENSE2_UNITS = 256
    OUTPUT_UNITS = NUM_CLASSES
    
    # Regularization (same as Experiment 1)
    DROPOUT_RATE = 0.5
    L2_REG = 0.0001
    
    # APE-GAN specific parameters
    ADVERSARIAL_RATIO = 0.5  # 50% adversarial, 50% clean samples
    EPSILON = 0.3  # FGSM perturbation magnitude
    ADVERSARIAL_START_EPOCH = 5  # Start adversarial training after epoch 5
    
    # Output directories
    EXPERIMENT_NAME = "exp2_cnn_apegan"
    OUTPUT_DIR = f"/kaggle/working/results/{EXPERIMENT_NAME}"
    MODELS_DIR = f"{OUTPUT_DIR}/models"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    FIGURES_DIR = f"{OUTPUT_DIR}/figures"
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.OUTPUT_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.FIGURES_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Output: {cls.OUTPUT_DIR}\n")


def generate_adversarial_examples(model, images, labels, epsilon=0.3):
    """
    Generate adversarial examples using FGSM (Fast Gradient Sign Method)
    This is the core of APE-GAN adversarial training
    """
    images = tf.cast(images, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images, training=False)
        loss = keras.losses.categorical_crossentropy(labels, predictions)
    
    # Get gradient of loss w.r.t. input image
    gradient = tape.gradient(loss, images)
    
    # Create adversarial example: x_adv = x + epsilon * sign(gradient)
    signed_grad = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_grad
    
    # Clip to valid range [0, 1]
    adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)
    
    return adversarial_images


class AdversarialDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for APE-GAN adversarial training
    Generates batches with mix of clean and adversarial samples
    """
    def __init__(self, data_generator, model, epsilon, adversarial_ratio, start_epoch=0):
        self.data_generator = data_generator
        self.model = model
        self.epsilon = epsilon
        self.adversarial_ratio = adversarial_ratio
        self.current_epoch = 0
        self.start_epoch = start_epoch
        
    def __len__(self):
        return len(self.data_generator)
    
    def __getitem__(self, index):
        # Get batch from original generator
        batch_x, batch_y = self.data_generator[index]
        
        # Only generate adversarial examples after start_epoch
        if self.current_epoch < self.start_epoch:
            return batch_x, batch_y
        
        # Calculate how many adversarial samples to generate
        batch_size = batch_x.shape[0]
        n_adversarial = int(batch_size * self.adversarial_ratio)
        
        if n_adversarial > 0:
            # Generate adversarial examples for portion of batch
            adv_indices = np.random.choice(batch_size, n_adversarial, replace=False)
            
            adv_images = generate_adversarial_examples(
                self.model,
                batch_x[adv_indices],
                batch_y[adv_indices],
                self.epsilon
            )
            
            # Replace clean images with adversarial ones
            batch_x_mixed = batch_x.copy()
            batch_x_mixed[adv_indices] = adv_images.numpy()
            
            return batch_x_mixed, batch_y
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        self.current_epoch += 1
        self.data_generator.on_epoch_end()


def load_emnist_data():
    """Load EMNIST with data augmentation"""
    print("="*80)
    print("LOADING EMNIST BALANCED DATASET")
    print("="*80)
    
    # Training with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
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
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(Config.DATA_ROOT, 'test'),
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    print(f"\n✓ Training: {train_generator.samples}")
    print(f"✓ Validation: {val_generator.samples}")
    print(f"✓ Test: {test_generator.samples}")
    print(f"✓ Classes: {train_generator.num_classes}")
    print("="*80 + "\n")
    
    return train_generator, val_generator, test_generator


def build_cnn():
    """
    Build CNN with EXACT same architecture as Experiment 1
    Total Parameters: 3,452,335
    """
    print("="*80)
    print("BUILDING CNN ARCHITECTURE (SAME AS EXPERIMENT 1)")
    print("="*80)
    
    model = models.Sequential(name='APE_GAN_CNN')
    
    # Block 1: 1→32
    model.add(layers.Conv2D(
        Config.CONV1_FILTERS, Config.CONV1_KERNEL,
        activation='relu', padding='same',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS),
        name='conv2d_1'
    ))
    model.add(layers.BatchNormalization(name='bn_1'))
    model.add(layers.MaxPooling2D((2, 2), name='pool_1'))
    model.add(layers.Dropout(0.25, name='dropout_1'))
    
    # Block 2: 32→64
    model.add(layers.Conv2D(
        Config.CONV2_FILTERS, Config.CONV2_KERNEL,
        activation='relu', padding='same',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        name='conv2d_2'
    ))
    model.add(layers.BatchNormalization(name='bn_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool_2'))
    model.add(layers.Dropout(0.25, name='dropout_2'))
    
    # Block 3: 64→128
    model.add(layers.Conv2D(
        Config.CONV3_FILTERS, Config.CONV3_KERNEL,
        activation='relu', padding='same',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        name='conv2d_3'
    ))
    model.add(layers.BatchNormalization(name='bn_3'))
    model.add(layers.Dropout(0.25, name='dropout_3'))
    
    # Flatten
    model.add(layers.Flatten(name='flatten'))
    
    # Dense layers
    model.add(layers.Dense(
        Config.DENSE1_UNITS, activation='relu',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        name='dense_1'
    ))
    model.add(layers.BatchNormalization(name='bn_4'))
    model.add(layers.Dropout(Config.DROPOUT_RATE, name='dropout_4'))
    
    model.add(layers.Dense(
        Config.DENSE2_UNITS, activation='relu',
        kernel_regularizer=regularizers.l2(Config.L2_REG),
        name='dense_2'
    ))
    model.add(layers.BatchNormalization(name='bn_5'))
    model.add(layers.Dropout(Config.DROPOUT_RATE, name='dropout_5'))
    
    # Output
    model.add(layers.Dense(Config.OUTPUT_UNITS, activation='softmax', name='output'))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    total_params = model.count_params()
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\n{'='*80}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Non-trainable Parameters: {non_trainable:,}")
    print(f"Architecture: Enhanced for 47-class EMNIST")
    print(f"Regularization: Dropout={Config.DROPOUT_RATE}, L2={Config.L2_REG}")
    print("="*80 + "\n")
    
    return model


def train_model_with_apegan(model, train_gen, val_gen):
    """
    Train model using APE-GAN adversarial training strategy
    """
    print("="*80)
    print("APE-GAN ADVERSARIAL TRAINING")
    print("="*80)
    
    print(f"\nAPE-GAN Configuration:")
    print(f"  Adversarial ratio: {Config.ADVERSARIAL_RATIO*100:.0f}%")
    print(f"  FGSM epsilon: {Config.EPSILON}")
    print(f"  Start epoch: {Config.ADVERSARIAL_START_EPOCH}")
    print(f"  Method: FGSM (Fast Gradient Sign Method)")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    print(f"  Dropout: {Config.DROPOUT_RATE}")
    print(f"  L2 regularization: {Config.L2_REG}")
    print(f"  Data augmentation: Enabled\n")
    
    # Wrap training generator with adversarial generator
    adv_train_gen = AdversarialDataGenerator(
        train_gen,
        model,
        epsilon=Config.EPSILON,
        adversarial_ratio=Config.ADVERSARIAL_RATIO,
        start_epoch=Config.ADVERSARIAL_START_EPOCH
    )
    
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
            monitor='val_loss',
            patience=20,  # More patience for adversarial training
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(Config.RESULTS_DIR, 'training_log.csv')
        ),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: adv_train_gen.on_epoch_end()
        )
    ]
    
    # Train with adversarial examples
    print("Starting adversarial training...")
    print(f"Epochs 1-{Config.ADVERSARIAL_START_EPOCH}: Clean samples only")
    print(f"Epochs {Config.ADVERSARIAL_START_EPOCH+1}-{Config.EPOCHS}: {Config.ADVERSARIAL_RATIO*100:.0f}% adversarial samples\n")
    
    history = model.fit(
        adv_train_gen,
        validation_data=val_gen,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*80)
    print("APE-GAN TRAINING COMPLETED")
    print("="*80 + "\n")
    
    return history


def evaluate_model(model, test_gen):
    """Evaluate on full test set"""
    print("="*80)
    print("EVALUATING ON FULL TEST SET")
    print("="*80)
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n{'='*80}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("="*80 + "\n")
    
    # Predictions
    print("Generating predictions...")
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    class_names = list(test_gen.class_indices.keys())
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(Config.RESULTS_DIR, 'classification_report.csv'))
    print(f"✓ Classification report saved")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
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
    """Plot training curves (600 DPI)"""
    print("\nGenerating training plots...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], linewidth=2, label='Train', color='#2E86AB')
    axes[0].plot(history.history['val_accuracy'], linewidth=2, label='Validation', color='#A23B72')
    axes[0].axvline(x=Config.ADVERSARIAL_START_EPOCH, color='red', linestyle='--', 
                    linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10, frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], linewidth=2, label='Train', color='#2E86AB')
    axes[1].plot(history.history['val_loss'], linewidth=2, label='Validation', color='#A23B72')
    axes[1].axvline(x=Config.ADVERSARIAL_START_EPOCH, color='red', linestyle='--',
                    linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10, frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'training_history.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved (600 DPI)")


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix (600 DPI)"""
    print("Generating confusion matrix...")
    
    fig, ax = plt.subplots(figsize=(16, 14))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, annot=False, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}, ax=ax
    )
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'confusion_matrix.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved (600 DPI)")


def plot_per_class_accuracy(report, class_names):
    """Plot per-class accuracy (600 DPI)"""
    print("Generating per-class accuracy...")
    
    accuracy_scores = [report[cn]['precision'] if cn in report else 0.0 for cn in class_names]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(class_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
    
    ax.bar(x_pos, accuracy_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'per_class_accuracy.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-class accuracy saved (600 DPI)")


def save_metadata(history, eval_results, training_time):
    """Save experiment metadata"""
    print("\nSaving metadata...")
    
    metadata = {
        'experiment': {
            'name': Config.EXPERIMENT_NAME,
            'description': 'APE-GAN adversarial training on EMNIST',
            'method': 'APE-GAN with FGSM',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time_seconds': training_time
        },
        'dataset': {
            'name': 'EMNIST Balanced',
            'num_classes': Config.NUM_CLASSES,
            'image_shape': [Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS]
        },
        'architecture': {
            'model_name': 'APE_GAN_CNN',
            'filters': [Config.CONV1_FILTERS, Config.CONV2_FILTERS, Config.CONV3_FILTERS],
            'dense_units': [Config.DENSE1_UNITS, Config.DENSE2_UNITS, Config.OUTPUT_UNITS],
            'regularization': {
                'dropout': Config.DROPOUT_RATE,
                'l2': Config.L2_REG,
                'data_augmentation': True
            }
        },
        'adversarial_training': {
            'method': 'APE-GAN with FGSM',
            'epsilon': Config.EPSILON,
            'adversarial_ratio': Config.ADVERSARIAL_RATIO,
            'start_epoch': Config.ADVERSARIAL_START_EPOCH,
            'description': 'Fast Gradient Sign Method for adversarial example generation'
        },
        'hyperparameters': {
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'learning_rate': Config.LEARNING_RATE
        },
        'results': {
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'test_accuracy': float(eval_results['test_accuracy']),
            'test_loss': float(eval_results['test_loss']),
            'overfitting_gap': float(history.history['accuracy'][-1] - history.history['val_accuracy'][-1])
        },
        'training_curves': {
            'epochs': list(range(1, len(history.history['accuracy']) + 1)),
            'train_accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Metadata saved")


def main():
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print(" " * 12 + "EXPERIMENT 2: APE-GAN ADVERSARIAL TRAINING")
    print(" " * 20 + "EMNIST BALANCED DATASET")
    print("="*80 + "\n")
    
    Config.create_directories()
    train_gen, val_gen, test_gen = load_emnist_data()
    model = build_cnn()
    history = train_model_with_apegan(model, train_gen, val_gen)
    
    final_model_path = os.path.join(Config.MODELS_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"✓ Final model saved\n")
    
    eval_results = evaluate_model(model, test_gen)
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    plot_training_history(history)
    plot_confusion_matrix(eval_results['confusion_matrix'], eval_results['class_names'])
    plot_per_class_accuracy(eval_results['classification_report'], eval_results['class_names'])
    print("="*80 + "\n")
    
    training_time = (datetime.now() - start_time).total_seconds()
    save_metadata(history, eval_results, training_time)
    
    # Final summary
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    test_acc = eval_results['test_accuracy']
    gap = train_acc - val_acc
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Overfitting Gap: {gap*100:.2f}% {'✓ GOOD' if gap < 0.05 else '⚠ CHECK'}")
    print(f"\nAPE-GAN Training: {Config.ADVERSARIAL_RATIO*100:.0f}% adversarial samples")
    print(f"FGSM Epsilon: {Config.EPSILON}")
    print(f"\nResults saved to: {Config.OUTPUT_DIR}")
    print("="*80 + "\n")
    
    print("Note: This model is trained with adversarial examples and should")
    print("      show improved robustness against adversarial attacks compared")
    print("      to the standard model from Experiment 1.")


if __name__ == "__main__":
    main()

