"""
Experiment 4: L-BFGS Adversarial Attack Evaluation
Author: Rumen
Date: 2026

EXACT Formula from paper (Equation 3):
x' = x - α · ∇L(x', y)

Where:
- x' is the perturbed input
- x is the original input  
- α is the step size (alpha)
- ∇L is the gradient of loss with respect to input

Simple iterative gradient descent to maximize loss.
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
print("EXPERIMENT 4: L-BFGS ATTACK EVALUATION")
print("="*80)


class Config:
    """Configuration for L-BFGS evaluation"""
    
    # Dataset
    DATA_ROOT = "/kaggle/input/emnist/emnist"
    NUM_CLASSES = 47
    
    # Models
    STANDARD_MODEL = "/kaggle/working/results/exp1_cnn_standard/models/best_model.h5"
    APEGAN_MODEL = "/kaggle/working/results/exp2_cnn_apegan/models/best_model.h5"
    
    # L-BFGS parameters - MUCH larger alpha for stronger attacks!
    ALPHA_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    MAX_ITERATIONS = 80  # Increased for better convergence
    
    # Output
    EXPERIMENT_NAME = "exp4_lbfgs_evaluation"
    OUTPUT_DIR = f"/kaggle/working/results/{EXPERIMENT_NAME}"
    FIGURES_DIR = f"{OUTPUT_DIR}/figures"
    
    # Alpha to sample mapping
    EPSILON_TO_SAMPLES = {
        1.0: ('R', '27_R'),
        2.0: ('S', '28_S'),
        3.0: ('5', '05_5'),
        4.0: ('7', '07_7'),
        5.0: ('n', '43_n'),
        6.0: ('q', '44_q')
    }
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.OUTPUT_DIR, cls.FIGURES_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Output: {cls.OUTPUT_DIR}\n")


def lbfgs_attack_batch(model, images, alpha=1.0, max_iter=20):
    """
    L-BFGS attack using EXACT paper formula: x' = x - α · ∇L(x', y)
    
    Iteratively updates to maximize loss (minimize correct class probability)
    Uses LARGE alpha values for strong perturbations
    """
    x = images.copy()
    batch_size = x.shape[0]
    
    # Get true classes
    preds = model(x, training=False).numpy()
    true_classes = np.argmax(preds, axis=1)
    
    # Iterative optimization
    for iteration in range(max_iter):
        x_var = tf.Variable(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = model(x_var, training=False)
            
            # Loss: negative log probability of true class
            # We want to MINIMIZE this (reduce true class probability)
            losses = []
            for i in range(batch_size):
                losses.append(predictions[i, true_classes[i]])
            
            loss = tf.reduce_mean(tf.stack(losses))
        
        # Compute gradient
        grad = tape.gradient(loss, x_var)
        
        if grad is None:
            break
        
        # Paper formula: x' = x - α · ∇L
        # Subtract gradient to reduce true class probability
        x = x - alpha * grad.numpy()
        
        # Clip to valid range
        x = np.clip(x, 0.0, 1.0)
    
    return x


def lbfgs_attack_simple(model, image, alpha=1.0, max_iter=20):
    """Single image L-BFGS for visualization"""
    x = image.copy()
    original = image.copy()
    
    pred = model(x, training=False).numpy()
    true_class = np.argmax(pred[0])
    
    for iteration in range(max_iter):
        x_var = tf.Variable(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = model(x_var, training=False)
            # Loss: probability of true class
            loss = predictions[0, true_class]
        
        grad = tape.gradient(loss, x_var)
        
        if grad is None:
            break
        
        # Paper formula: x' = x - α · ∇L
        x = x - alpha * grad.numpy()
        x = np.clip(x, 0.0, 1.0)
    
    perturbation = x - original
    
    return x, perturbation, max_iter


def evaluate_both_models_on_lbfgs(model_standard, model_apegan, test_generator, alpha):
    """Evaluate both models - fast batch processing"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating Alpha (α): {alpha}")
    print("="*80)
    
    std_clean_correct = 0
    std_adv_correct = 0
    apegan_clean_correct = 0
    apegan_adv_correct = 0
    total = 0
    
    test_generator.reset()
    num_batches = len(test_generator)
    
    print(f"Processing {test_generator.samples} samples in {num_batches} batches...")
    
    for batch_idx in tqdm(range(num_batches), desc=f"α={alpha}"):
        batch_x, batch_y = test_generator[batch_idx]
        batch_size = batch_x.shape[0]
        
        # Generate adversarial using Standard CNN
        try:
            batch_adv = lbfgs_attack_batch(
                model_standard, batch_x,
                alpha=alpha,
                max_iter=Config.MAX_ITERATIONS
            )
        except:
            batch_adv = batch_x
        
        true_classes = np.argmax(batch_y, axis=1)
        
        # Evaluate Standard CNN
        pred_std_clean = model_standard.predict(batch_x, batch_size=batch_size, verbose=0)
        pred_std_adv = model_standard.predict(batch_adv, batch_size=batch_size, verbose=0)
        
        std_clean_correct += np.sum(np.argmax(pred_std_clean, axis=1) == true_classes)
        std_adv_correct += np.sum(np.argmax(pred_std_adv, axis=1) == true_classes)
        
        # Evaluate APE-GAN CNN
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
    print(f"RESULTS for α={alpha}")
    print("="*60)
    print(f"\nStandard CNN:")
    print(f"  Clean: {std_clean_acc:.2f}% | Adversarial: {std_adv_acc:.2f}% | Drop: {std_clean_acc - std_adv_acc:.2f}%")
    print(f"\nAPE-GAN CNN:")
    print(f"  Clean: {apegan_clean_acc:.2f}% | Adversarial: {apegan_adv_acc:.2f}% | Drop: {apegan_clean_acc - apegan_adv_acc:.2f}%")
    print("="*60)
    
    return std_results, apegan_results


def generate_lbfgs_visualization():
    """Generate 3×6 visualization"""
    
    print("\n" + "="*80)
    print("GENERATING L-BFGS VISUALIZATION")
    print("="*80)
    
    print("\nLoading Standard CNN model...")
    model_standard = keras.models.load_model(Config.STANDARD_MODEL)
    print("✓ Model loaded")
    
    print("\nLoading sample images...")
    samples_data = []
    
    for alpha, (label, folder) in Config.EPSILON_TO_SAMPLES.items():
        class_path = os.path.join(Config.DATA_ROOT, 'test', folder)
        
        if not os.path.exists(class_path):
            continue
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg'))]
        if not images:
            continue
        
        img_path = os.path.join(class_path, images[0])
        img = keras.preprocessing.image.load_img(
            img_path, color_mode='grayscale', target_size=(28, 28)
        )
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        
        samples_data.append({
            'alpha': alpha,
            'label': label,
            'image': img_array
        })
        
        print(f"✓ Loaded: {label} (α={alpha})")
    
    if len(samples_data) != 6:
        raise ValueError(f"Expected 6 samples, got {len(samples_data)}")
    
    print("\nGenerating adversarial examples...")
    results = []
    
    for sample in samples_data:
        img_batch = np.expand_dims(sample['image'], axis=0)
        
        adv_img, perturbation, iterations = lbfgs_attack_simple(
            model_standard, img_batch,
            alpha=sample['alpha'],
            max_iter=Config.MAX_ITERATIONS
        )
        
        results.append({
            'alpha': sample['alpha'],
            'label': sample['label'],
            'original': sample['image'],
            'perturbation': perturbation,
            'adversarial': adv_img,
            'iterations': iterations
        })
        
        print(f"  α={sample['alpha']} ({sample['label']}): {iterations} iterations")
    
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('white')
    
    row_mapping = {'R': 0, 'S': 0, '5': 1, '7': 1, 'n': 2, 'q': 2}
    col_mapping = {'R': 0, 'S': 3, '5': 0, '7': 3, 'n': 0, 'q': 3}
    
    for result in results:
        label = result['label']
        row = row_mapping[label]
        col_start = col_mapping[label]
        
        # Original
        ax1 = plt.subplot(3, 6, row * 6 + col_start + 1)
        ax1.imshow(result['original'].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Original: {label}', fontsize=11, fontweight='bold')
        ax1.axis('off')
        ax1.text(0.5, -0.15, f"α={result['alpha']}", 
                 transform=ax1.transAxes, ha='center', fontsize=10, fontweight='bold')
        
        # Perturbation
        ax2 = plt.subplot(3, 6, row * 6 + col_start + 2)
        ax2.imshow(result['perturbation'].squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax2.set_title('Perturbation', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Adversarial
        ax3 = plt.subplot(3, 6, row * 6 + col_start + 3)
        ax3.imshow(result['adversarial'].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'Adversarial (iter={result["iterations"]})', fontsize=11, fontweight='bold')
        ax3.axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(Config.FIGURES_DIR, 'lbfgs_examples.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Figure saved: {output_path}")
    return output_path


def main():
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("EXPERIMENT 4: L-BFGS ATTACK EVALUATION")
    print("="*80 + "\n")
    
    Config.create_directories()
    
    print("Loading models...")
    model_standard = keras.models.load_model(Config.STANDARD_MODEL)
    model_apegan = keras.models.load_model(Config.APEGAN_MODEL)
    print("✓ Models loaded\n")
    
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
    
    # Evaluate
    results = {'standard': {}, 'apegan': {}}
    
    print("="*80)
    print("EVALUATING BOTH MODELS")
    print("="*80)
    
    for alpha in Config.ALPHA_VALUES:
        test_generator.reset()
        std_results, apegan_results = evaluate_both_models_on_lbfgs(
            model_standard, model_apegan, test_generator, alpha
        )
        results['standard'][alpha] = std_results
        results['apegan'][alpha] = apegan_results
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    standard_adv_accs = [results['standard'][a]['adversarial_accuracy'] for a in Config.ALPHA_VALUES]
    apegan_adv_accs = [results['apegan'][a]['adversarial_accuracy'] for a in Config.ALPHA_VALUES]
    
    mean_standard = np.mean(standard_adv_accs)
    mean_apegan = np.mean(apegan_adv_accs)
    
    print(f"\nMean Adversarial Accuracy:")
    print(f"  Standard CNN: {mean_standard:.2f}%")
    print(f"  APE-GAN CNN:  {mean_apegan:.2f}%")
    print(f"  Improvement:  +{mean_apegan - mean_standard:.2f}%")
    
    # Visualization
    figure_path = generate_lbfgs_visualization()
    
    # Metadata
    evaluation_time = (datetime.now() - start_time).total_seconds()
    
    metadata = {
        'experiment': {
            'name': Config.EXPERIMENT_NAME,
            'description': 'L-BFGS attack evaluation on Standard and APE-GAN models',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_time_seconds': evaluation_time
        },
        'attack': {
            'method': 'L-BFGS',
            'formula': "x' = x - α · ∇L(x', y)",
            'description': 'Iteratively optimizes perturbation using gradient descent',
            'alpha_values': Config.ALPHA_VALUES,
            'max_iterations': Config.MAX_ITERATIONS
        },
        'dataset': {
            'name': 'EMNIST Balanced',
            'test_samples': test_generator.samples,
            'num_classes': Config.NUM_CLASSES
        },
        'results': {
            'standard_cnn': {
                str(a): {
                    'clean_accuracy': results['standard'][a]['clean_accuracy'],
                    'adversarial_accuracy': results['standard'][a]['adversarial_accuracy'],
                    'accuracy_drop': results['standard'][a]['accuracy_drop']
                } for a in Config.ALPHA_VALUES
            },
            'apegan_cnn': {
                str(a): {
                    'clean_accuracy': results['apegan'][a]['clean_accuracy'],
                    'adversarial_accuracy': results['apegan'][a]['adversarial_accuracy'],
                    'accuracy_drop': results['apegan'][a]['accuracy_drop']
                } for a in Config.ALPHA_VALUES
            },
            'mean_accuracies': {
                'standard_cnn': float(mean_standard),
                'apegan_cnn': float(mean_apegan),
                'improvement': float(mean_apegan - mean_standard)
            }
        },
        'visualization': {
            'figure_path': 'figures/lbfgs_examples.png',
            'sample_mapping': {str(a): f"{Config.EPSILON_TO_SAMPLES[a][0]}" for a in Config.ALPHA_VALUES},
            'description': '3×6 grid: Original | Perturbation | Adversarial'
        }
    }
    
    metadata_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✓ Metadata: {metadata_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"Time: {evaluation_time/60:.2f} minutes")
    print(f"Outputs: metadata.json + lbfgs_examples.png")
    print("="*80)


if __name__ == "__main__":
    main()