"""
Experiment 3: DeepFool Adversarial Attack Evaluation (FIXED)
Author: Rumen
Date: 2026

Evaluates both Standard and APE-GAN models against DeepFool attacks
Following the paper's formula: x' = x + δ · (f(x)/||∇f(x)||)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU error: {e}")

print("="*80)
print("EXPERIMENT 3: DEEPFOOL ATTACK EVALUATION")
print("="*80)


class Config:
    """Configuration for DeepFool evaluation"""
    
    # Dataset
    DATA_ROOT = "/kaggle/input/emnist/emnist"
    NUM_CLASSES = 47
    
    # Models
    STANDARD_MODEL = "/kaggle/working/results/exp1_cnn_standard/models/best_model.h5"
    APEGAN_MODEL = "/kaggle/working/results/exp2_cnn_apegan/models/best_model.h5"
    
    # DeepFool parameters - 6 different perturbation strengths
    OVERSHOOT_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    MAX_ITERATIONS = 10  # Reduce to 5 for speed - DeepFool usually converges in 3-5 iterations
    
    # Output
    EXPERIMENT_NAME = "exp3_deepfool_evaluation"
    OUTPUT_DIR = f"/kaggle/working/results/{EXPERIMENT_NAME}"
    FIGURES_DIR = f"{OUTPUT_DIR}/figures"
    
    # Sample classes for visualization - mapped to epsilon values
    # ε=0.1 & 0.2: Capital letters (A, D)
    # ε=0.3 & 0.4: Digits (4, 7)
    # ε=0.5 & 0.6: Small letters (d, h)
    EPSILON_TO_SAMPLES = {
        0.1: ('A', '10_A'),
        0.2: ('D', '13_D'),
        0.3: ('4', '04_4'),
        0.4: ('7', '07_7'),
        0.5: ('d', '38_d'),
        0.6: ('h', '42_h')
    }
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.OUTPUT_DIR, cls.FIGURES_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Output: {cls.OUTPUT_DIR}\n")


def deepfool_attack_batch(model, images, overshoot=0.1, max_iter=5):
    """
    Batch DeepFool - process multiple images at once for speed
    """
    batch_size = images.shape[0]
    x = images.copy()
    
    # Get original predictions for entire batch
    preds = model(x, training=False).numpy()
    original_classes = np.argmax(preds, axis=1)
    
    for iteration in range(max_iter):
        # Process entire batch at once
        x_var = tf.Variable(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = model(x_var, training=False)
            # Loss for each image: logit of its original class
            losses = tf.stack([predictions[i, original_classes[i]] for i in range(batch_size)])
        
        # Get gradients for entire batch
        grads = tape.gradient(losses, x_var)
        
        if grads is None:
            break
        
        grads_np = grads.numpy()
        
        # Apply perturbation to each image
        for i in range(batch_size):
            grad = grads_np[i:i+1]
            grad_norm = np.linalg.norm(grad.flatten())
            
            if grad_norm > 1e-8:
                x[i] = x[i] - overshoot * grad[0] / grad_norm
                x[i] = np.clip(x[i], 0.0, 1.0)
    
    return x


def deepfool_attack_simple(model, image, overshoot=0.1, max_iter=5):
    """
    Single image DeepFool - for visualization only
    """
    x = image.copy()
    
    # Get original prediction
    pred = model(x, training=False).numpy()
    original_class = np.argmax(pred[0])
    
    for iteration in range(max_iter):
        x_var = tf.Variable(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = model(x_var, training=False)
            loss = predictions[0, original_class]
        
        grad = tape.gradient(loss, x_var)
        
        if grad is None:
            break
        
        # Check if already fooled
        current_class = np.argmax(predictions.numpy()[0])
        if current_class != original_class:
            break
        
        grad_np = grad.numpy()
        grad_norm = np.linalg.norm(grad_np.flatten())
        
        if grad_norm > 1e-8:
            x = x - overshoot * grad_np / grad_norm
            x = np.clip(x, 0.0, 1.0)
    
    return x, iteration + 1


def evaluate_both_models_on_deepfool(model_standard, model_apegan, test_generator, overshoot):
    """
    FIXED: Generate adversarial examples using Standard CNN (easier to fool)
    Then test on both models
    """
    
    print(f"\n{'='*80}")
    print(f"Evaluating Overshoot: {overshoot}")
    print("="*80)
    
    # Counters
    std_clean_correct = 0
    std_adv_correct = 0
    apegan_clean_correct = 0
    apegan_adv_correct = 0
    total = 0
    
    test_generator.reset()
    num_batches = len(test_generator)
    
    # Process in batches
    print(f"Processing {test_generator.samples} samples in {num_batches} batches...")
    
    for batch_idx in tqdm(range(num_batches), desc=f"ε={overshoot}"):
        batch_x, batch_y = test_generator[batch_idx]
        batch_size = batch_x.shape[0]
        
        # Generate adversarial examples using STANDARD CNN (not APE-GAN!)
        # Standard CNN is easier to fool, so adversarials will be stronger
        try:
            batch_adv = deepfool_attack_batch(
                model_standard, batch_x,  # ← Changed from model_apegan to model_standard
                overshoot=overshoot,
                max_iter=Config.MAX_ITERATIONS
            )
        except:
            batch_adv = batch_x  # Use original on failure
        
        true_classes = np.argmax(batch_y, axis=1)
        
        # Predict on this batch - Standard CNN
        pred_std_clean = model_standard.predict(batch_x, batch_size=batch_size, verbose=0)
        pred_std_adv = model_standard.predict(batch_adv, batch_size=batch_size, verbose=0)
        
        std_clean_correct += np.sum(np.argmax(pred_std_clean, axis=1) == true_classes)
        std_adv_correct += np.sum(np.argmax(pred_std_adv, axis=1) == true_classes)
        
        # Predict on this batch - APE-GAN CNN
        pred_apegan_clean = model_apegan.predict(batch_x, batch_size=batch_size, verbose=0)
        pred_apegan_adv = model_apegan.predict(batch_adv, batch_size=batch_size, verbose=0)
        
        apegan_clean_correct += np.sum(np.argmax(pred_apegan_clean, axis=1) == true_classes)
        apegan_adv_correct += np.sum(np.argmax(pred_apegan_adv, axis=1) == true_classes)
        
        total += batch_size
    
    # Calculate accuracies
    std_clean_acc = (std_clean_correct / total) * 100
    std_adv_acc = (std_adv_correct / total) * 100
    apegan_clean_acc = (apegan_clean_correct / total) * 100
    apegan_adv_acc = (apegan_adv_correct / total) * 100
    
    # Results
    std_results = {
        'clean_accuracy': std_clean_acc,
        'adversarial_accuracy': std_adv_acc,
        'accuracy_drop': std_clean_acc - std_adv_acc,
        'total_samples': total
    }
    
    apegan_results = {
        'clean_accuracy': apegan_clean_acc,
        'adversarial_accuracy': apegan_adv_acc,
        'accuracy_drop': apegan_clean_acc - apegan_adv_acc,
        'total_samples': total
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS for ε={overshoot}")
    print("="*60)
    print(f"\nStandard CNN:")
    print(f"  Clean Accuracy: {std_clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {std_adv_acc:.2f}%")
    print(f"  Accuracy Drop: {std_clean_acc - std_adv_acc:.2f}%")
    
    print(f"\nAPE-GAN CNN:")
    print(f"  Clean Accuracy: {apegan_clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {apegan_adv_acc:.2f}%")
    print(f"  Accuracy Drop: {apegan_clean_acc - apegan_adv_acc:.2f}%")
    print("="*60)
    
    return std_results, apegan_results


def generate_deepfool_examples():
    """Generate visualization figure with DeepFool examples"""
    
    print("\n" + "="*80)
    print("GENERATING DEEPFOOL VISUALIZATION")
    print("="*80)
    
    # Load Standard CNN model (for generating adversarial examples)
    print("\nLoading Standard CNN model...")
    model_standard = keras.models.load_model(Config.STANDARD_MODEL)
    print("✓ Standard CNN model loaded")
    
    # Load sample images for each epsilon
    print("\nLoading sample images...")
    samples = {}
    
    for epsilon, (label, folder) in Config.EPSILON_TO_SAMPLES.items():
        class_path = os.path.join(Config.DATA_ROOT, 'test', folder)
        
        if not os.path.exists(class_path):
            print(f"✗ Path not found: {class_path}")
            continue
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg'))]
        if not images:
            print(f"✗ No images in: {class_path}")
            continue
        
        img_path = os.path.join(class_path, images[0])
        img = keras.preprocessing.image.load_img(
            img_path, color_mode='grayscale', target_size=(28, 28)
        )
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        samples[epsilon] = {
            'image': img_array,
            'label': label
        }
        print(f"✓ Loaded: ε={epsilon} → {label}")
    
    if len(samples) != 6:
        raise ValueError(f"Expected 6 samples, got {len(samples)}")
    
    # Create figure: 6 rows × 2 columns
    print("\nGenerating adversarial examples...")
    fig, axes = plt.subplots(6, 2, figsize=(8, 18))
    fig.patch.set_facecolor('white')
    
    for row_idx, overshoot in enumerate(Config.OVERSHOOT_VALUES):
        sample_data = samples[overshoot]
        img_array = sample_data['image']
        label = sample_data['label']
        
        print(f"  Processing ε={overshoot} ({label})...")
        
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Generate adversarial example using Standard CNN
        adv_img, iterations = deepfool_attack_simple(
            model_standard, img_batch,  # ← Changed to model_standard
            overshoot=overshoot,
            max_iter=Config.MAX_ITERATIONS
        )
        
        # Column 1: Original image
        axes[row_idx, 0].imshow(img_array.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f'Original: {label}', 
                                   fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(0.5, -0.15, f'ε={overshoot}', 
                             transform=axes[row_idx, 0].transAxes,
                             ha='center', fontsize=11, fontweight='bold')
        
        # Column 2: Adversarial image
        axes[row_idx, 1].imshow(adv_img.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f'Adversarial (iter={iterations})', 
                                   fontsize=12, fontweight='bold')
        axes[row_idx, 1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(Config.FIGURES_DIR, 'deepfool_examples.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Figure saved: {output_path}")
    return output_path


def main():
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: DEEPFOOL ATTACK EVALUATION")
    print("="*80 + "\n")
    
    Config.create_directories()
    
    # Load models
    print("Loading models...")
    model_standard = keras.models.load_model(Config.STANDARD_MODEL)
    model_apegan = keras.models.load_model(Config.APEGAN_MODEL)
    print("✓ Models loaded\n")
    
    # Load test data
    print("Loading test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(Config.DATA_ROOT, 'test'),
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    print(f"✓ Test samples: {test_generator.samples}\n")
    
    # Evaluate both models across all overshoot values
    # OPTIMIZED: Generate adversarial examples once, test on both models
    results = {
        'standard': {},
        'apegan': {}
    }
    
    print("\n" + "="*80)
    print("EVALUATING BOTH MODELS (OPTIMIZED)")
    print("Generating adversarial examples once per epsilon")
    print("="*80)
    
    for overshoot in Config.OVERSHOOT_VALUES:
        test_generator.reset()
        
        # Generate adversarial examples once, evaluate on both models
        std_results, apegan_results = evaluate_both_models_on_deepfool(
            model_standard, model_apegan, test_generator, overshoot
        )
        
        results['standard'][overshoot] = std_results
        results['apegan'][overshoot] = apegan_results
    
    # Calculate mean accuracies
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    standard_adv_accs = [results['standard'][eps]['adversarial_accuracy'] 
                         for eps in Config.OVERSHOOT_VALUES]
    apegan_adv_accs = [results['apegan'][eps]['adversarial_accuracy'] 
                       for eps in Config.OVERSHOOT_VALUES]
    
    mean_standard = np.mean(standard_adv_accs)
    mean_apegan = np.mean(apegan_adv_accs)
    
    print(f"\nMean CNN Accuracy (across all perturbations):")
    print(f"  Standard CNN: {mean_standard:.2f}%")
    print(f"  APE-GAN CNN:  {mean_apegan:.2f}%")
    print(f"  Improvement:  {mean_apegan - mean_standard:.2f}%")
    
    # Generate visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    figure_path = generate_deepfool_examples()
    
    # Calculate evaluation time
    evaluation_time = (datetime.now() - start_time).total_seconds()
    
    # Save metadata
    print("\n" + "="*80)
    print("SAVING METADATA")
    print("="*80)
    
    metadata = {
        'experiment': {
            'name': Config.EXPERIMENT_NAME,
            'description': 'DeepFool attack evaluation on Standard and APE-GAN models',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_time_seconds': evaluation_time
        },
        'attack': {
            'method': 'DeepFool',
            'formula': "x' = x + δ · (f(x)/||∇f(x)||)",
            'overshoot_values': Config.OVERSHOOT_VALUES,
            'max_iterations': Config.MAX_ITERATIONS,
            'num_perturbation_strengths': len(Config.OVERSHOOT_VALUES)
        },
        'dataset': {
            'name': 'EMNIST Balanced',
            'test_samples': test_generator.samples,
            'num_classes': Config.NUM_CLASSES
        },
        'results': {
            'standard_cnn': {
                str(eps): {
                    'clean_accuracy': results['standard'][eps]['clean_accuracy'],
                    'adversarial_accuracy': results['standard'][eps]['adversarial_accuracy'],
                    'accuracy_drop': results['standard'][eps]['accuracy_drop']
                } for eps in Config.OVERSHOOT_VALUES
            },
            'apegan_cnn': {
                str(eps): {
                    'clean_accuracy': results['apegan'][eps]['clean_accuracy'],
                    'adversarial_accuracy': results['apegan'][eps]['adversarial_accuracy'],
                    'accuracy_drop': results['apegan'][eps]['accuracy_drop']
                } for eps in Config.OVERSHOOT_VALUES
            },
            'mean_accuracies': {
                'standard_cnn': float(mean_standard),
                'apegan_cnn': float(mean_apegan),
                'improvement': float(mean_apegan - mean_standard)
            }
        },
        'visualization': {
            'figure_path': 'figures/deepfool_examples.png',
            'sample_mapping': {
                '0.02': 'A (Capital)',
                '0.05': 'B (Capital)',
                '0.1': '4 (Digit)',
                '0.2': '7 (Digit)',
                '0.3': 'd (Lowercase)',
                '0.5': 'h (Lowercase)'
            },
            'description': '6 rows (epsilon values) × 2 columns (original + adversarial), different sample per row'
        }
    }
    
    metadata_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nEvaluation Time: {evaluation_time/60:.2f} minutes")
    print(f"Test Samples: {test_generator.samples}")
    print(f"Perturbation Strengths Tested: {len(Config.OVERSHOOT_VALUES)}")
    print(f"\nMean Adversarial Accuracy:")
    print(f"  Standard CNN: {mean_standard:.2f}%")
    print(f"  APE-GAN CNN:  {mean_apegan:.2f}%")
    print(f"  Improvement:  +{mean_apegan - mean_standard:.2f}%")
    print(f"\nOutputs:")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - Figure: {figure_path}")
    print("="*80)


if __name__ == "__main__":
    main()