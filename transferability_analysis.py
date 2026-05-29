"""
Experiment 5: Adversarial Transferability Analysis
Merged DeepFool + L-BFGS transferability in a single experiment.

DeepFool  — δ ∈ {0.1, 0.3, 0.6},  iterations ∈ {5, 10}
L-BFGS    — α ∈ {1.0, 3.0, 6.0},  iterations ∈ {40, 80}

EXACT same attack functions as Experiment 3 (DeepFool) and Experiment 4 (L-BFGS).
No changes to formula, hyperparameters, batch size, generator settings.

DeepFool formula  (Exp 3 verbatim):
    x[i] = x[i] - overshoot * grad[0] / grad_norm
    x[i] = np.clip(x[i], 0.0, 1.0)

L-BFGS formula  (Exp 4 verbatim):
    x = x - alpha * grad.numpy()
    x = np.clip(x, 0.0, 1.0)

Direction 1 (Std→APE):
    Adversarial generated on Standard CNN.
    Results for FULL grids already exist in Exp 3 / Exp 4.
    Only the 3 selected parameter values are extracted — NOT recomputed.

Direction 2 (APE→Std):
    Adversarial generated on APE-GAN CNN, evaluated on Standard CNN.
    Newly computed in this experiment for the 3 selected parameter values.

No figure. Clean metadata.json for single research paper table.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU error: {e}")

print("=" * 80)
print("EXPERIMENT 5: ADVERSARIAL TRANSFERABILITY ANALYSIS")
print("DeepFool  δ ∈ {0.1, 0.3, 0.6}  ×  iter ∈ {5, 10}")
print("L-BFGS    α ∈ {1.0, 3.0, 6.0}  ×  iter ∈ {40, 80}")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Dataset — SAME AS EXP 3 / EXP 4
    DATA_ROOT   = "/kaggle/input/emnist/emnist"
    NUM_CLASSES = 47

    # Models — SAME PATHS AS EXP 3 / EXP 4
    STANDARD_MODEL = "/kaggle/working/results/exp1_cnn_standard/models/best_model.h5"
    APEGAN_MODEL   = "/kaggle/working/results/exp2_cnn_apegan/models/best_model.h5"

    # DeepFool — 3 selected values from Exp 3 full grid
    DF_OVERSHOOT_VALUES = [0.1, 0.3, 0.6]
    DF_ITERATIONS       = [5, 10]

    # L-BFGS — 3 selected values from Exp 4 full grid
    LB_ALPHA_VALUES  = [1.0, 3.0, 6.0]
    LB_ITERATIONS    = [40, 80]

    # Data loader — SAME AS EXP 3 / EXP 4
    BATCH_SIZE = 32

    # ── Direction 1 reference values from Exp 3 Table II
    # (adversarial from Standard CNN, evaluated on both models)
    EXP3_D1 = {
        5: {
            0.1: {'std_wb': 81.77, 'ape_trf': 88.12},
            0.3: {'std_wb': 46.06, 'ape_trf': 85.06},
            0.6: {'std_wb': 14.36, 'ape_trf': 80.68},
        },
        10: {
            0.1: {'std_wb': 61.13, 'ape_trf': 86.71},
            0.3: {'std_wb': 11.16, 'ape_trf': 80.51},
            0.6: {'std_wb':  4.84, 'ape_trf': 67.44},
        }
    }

    # ── Direction 1 reference values from Exp 4 Table III
    # (adversarial from Standard CNN, evaluated on both models)
    EXP4_D1 = {
        40: {
            1.0: {'std_wb': 74.41, 'ape_trf': 86.79},
            3.0: {'std_wb': 65.11, 'ape_trf': 85.32},
            6.0: {'std_wb': 58.58, 'ape_trf': 83.93},
        },
        80: {
            1.0: {'std_wb': 68.40, 'ape_trf': 85.90},
            3.0: {'std_wb': 58.18, 'ape_trf': 84.40},
            6.0: {'std_wb': 51.37, 'ape_trf': 83.07},
        }
    }

    # Clean accuracy from paper Table I
    STD_CLEAN_ACC    = 90.18
    APEGAN_CLEAN_ACC = 88.66

    # Output
    EXPERIMENT_NAME = "exp5_transferability"
    OUTPUT_DIR      = f"/kaggle/working/results/{EXPERIMENT_NAME}"
    RESULTS_DIR     = f"{OUTPUT_DIR}/results"

    @classmethod
    def create_directories(cls):
        for d in [cls.OUTPUT_DIR, cls.RESULTS_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directories ready: {cls.OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# DEEPFOOL ATTACK — VERBATIM FROM EXPERIMENT 3 (lines 80-117)
# ─────────────────────────────────────────────────────────────────────────────

def deepfool_attack_batch(model, images, overshoot=0.1, max_iter=5):
    """
    Batch DeepFool — COPIED VERBATIM FROM EXPERIMENT 3. Zero changes.
    """
    batch_size = images.shape[0]
    x = images.copy()

    preds = model(x, training=False).numpy()
    original_classes = np.argmax(preds, axis=1)

    for iteration in range(max_iter):
        x_var = tf.Variable(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            predictions = model(x_var, training=False)
            losses = tf.stack([predictions[i, original_classes[i]] for i in range(batch_size)])

        grads = tape.gradient(losses, x_var)

        if grads is None:
            break

        grads_np = grads.numpy()

        for i in range(batch_size):
            grad = grads_np[i:i+1]
            grad_norm = np.linalg.norm(grad.flatten())

            if grad_norm > 1e-8:
                x[i] = x[i] - overshoot * grad[0] / grad_norm   # Exp 3 line 114
                x[i] = np.clip(x[i], 0.0, 1.0)                  # Exp 3 line 115

    return x


# ─────────────────────────────────────────────────────────────────────────────
# L-BFGS ATTACK — VERBATIM FROM EXPERIMENT 4 (lines 85-127)
# ─────────────────────────────────────────────────────────────────────────────

def lbfgs_attack_batch(model, images, alpha=1.0, max_iter=40):
    """
    Batch L-BFGS — COPIED VERBATIM FROM EXPERIMENT 4. Zero changes.
    """
    x = images.copy()
    batch_size = x.shape[0]

    preds = model(x, training=False).numpy()
    true_classes = np.argmax(preds, axis=1)

    for iteration in range(max_iter):
        x_var = tf.Variable(x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            predictions = model(x_var, training=False)
            losses = []
            for i in range(batch_size):
                losses.append(predictions[i, true_classes[i]])
            loss = tf.reduce_mean(tf.stack(losses))

        grad = tape.gradient(loss, x_var)

        if grad is None:
            break

        x = x - alpha * grad.numpy()    # Exp 4 line 122
        x = np.clip(x, 0.0, 1.0)       # Exp 4 line 125

    return x


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTION 2 EVALUATION (APE→Std) — same structure as Exp 3 / Exp 4
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ape_to_std(model_apegan, model_standard, test_generator,
                        attack_fn, attack_kwargs, param_label):
    """
    Generates adversarial on APE-GAN CNN (white-box source),
    evaluates on both APE-GAN CNN (white-box) and Standard CNN (transfer).
    attack_fn   : deepfool_attack_batch or lbfgs_attack_batch
    attack_kwargs: dict of kwargs passed to attack_fn beyond model/images
    param_label : string for tqdm description
    """
    test_generator.reset()

    ape_clean_correct  = 0
    ape_adv_correct    = 0
    std_clean_correct  = 0
    std_trf_correct    = 0
    total = 0

    num_batches = len(test_generator)

    for batch_idx in tqdm(range(num_batches), desc=f"APE→Std {param_label}"):
        batch_x, batch_y = test_generator[batch_idx]
        batch_size = batch_x.shape[0]
        true_classes = np.argmax(batch_y, axis=1)

        try:
            batch_adv = attack_fn(model_apegan, batch_x, **attack_kwargs)
        except Exception as e:
            print(f"  Attack failed batch {batch_idx}: {e} — using clean")
            batch_adv = batch_x

        # APE-GAN clean + white-box adv
        pred_ape_clean = model_apegan.predict(batch_x,   batch_size=batch_size, verbose=0)
        pred_ape_adv   = model_apegan.predict(batch_adv, batch_size=batch_size, verbose=0)
        ape_clean_correct += np.sum(np.argmax(pred_ape_clean, axis=1) == true_classes)
        ape_adv_correct   += np.sum(np.argmax(pred_ape_adv,   axis=1) == true_classes)

        # Standard CNN clean + transfer adv
        pred_std_clean = model_standard.predict(batch_x,   batch_size=batch_size, verbose=0)
        pred_std_adv   = model_standard.predict(batch_adv, batch_size=batch_size, verbose=0)
        std_clean_correct += np.sum(np.argmax(pred_std_clean, axis=1) == true_classes)
        std_trf_correct   += np.sum(np.argmax(pred_std_adv,   axis=1) == true_classes)

        total += batch_size

    ape_clean = round(ape_clean_correct / total * 100, 2)
    ape_wb    = round(ape_adv_correct   / total * 100, 2)
    std_clean = round(std_clean_correct / total * 100, 2)
    std_trf   = round(std_trf_correct   / total * 100, 2)
    gap       = round(ape_wb - std_trf, 2)

    print(f"  APE-GAN white-box : {ape_wb}%  |  Standard transfer: {std_trf}%  |  Gap: {gap}%")

    return {
        'apegan_clean_accuracy':        ape_clean,
        'apegan_whitebox_adv_accuracy': ape_wb,
        'apegan_accuracy_drop':         round(ape_clean - ape_wb, 2),
        'standard_clean_accuracy':      std_clean,
        'standard_transfer_accuracy':   std_trf,
        'standard_accuracy_drop':       round(std_clean - std_trf, 2),
        'transferability_gap':          gap,
        'total_samples':                total
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("EXPERIMENT 5: ADVERSARIAL TRANSFERABILITY ANALYSIS")
    print("=" * 80 + "\n")

    Config.create_directories()

    print("Loading models...")
    model_standard = keras.models.load_model(Config.STANDARD_MODEL)
    model_apegan   = keras.models.load_model(Config.APEGAN_MODEL)
    print(f"✓ Standard CNN : {Config.STANDARD_MODEL}")
    print(f"✓ APE-GAN CNN  : {Config.APEGAN_MODEL}\n")

    # Data generator — SAME AS EXP 3 / EXP 4
    print("Loading test data...")
    test_datagen   = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(Config.DATA_ROOT, 'test'),
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    print(f"✓ Test samples: {test_generator.samples}\n")

    # ── Storage
    df_d2  = {it: {} for it in Config.DF_ITERATIONS}   # DeepFool Direction 2
    lb_d2  = {it: {} for it in Config.LB_ITERATIONS}   # L-BFGS   Direction 2

    total_configs = (len(Config.DF_ITERATIONS) * len(Config.DF_OVERSHOOT_VALUES) +
                     len(Config.LB_ITERATIONS) * len(Config.LB_ALPHA_VALUES))
    config_count = 0

    # ─────────────────────────────────────────────
    # DEEPFOOL Direction 2
    # ─────────────────────────────────────────────
    print("=" * 80)
    print("DEEPFOOL — Direction 2 (APE→Std)")
    print("=" * 80)

    for max_iter in Config.DF_ITERATIONS:
        for overshoot in Config.DF_OVERSHOOT_VALUES:
            config_count += 1
            print(f"\n[Config {config_count}/{total_configs}] "
                  f"DeepFool  δ={overshoot}  iter={max_iter}")
            test_generator.reset()
            df_d2[max_iter][overshoot] = evaluate_ape_to_std(
                model_apegan, model_standard, test_generator,
                attack_fn=deepfool_attack_batch,
                attack_kwargs={'overshoot': overshoot, 'max_iter': max_iter},
                param_label=f"δ={overshoot} iter={max_iter}"
            )

    # ─────────────────────────────────────────────
    # L-BFGS Direction 2
    # ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("L-BFGS — Direction 2 (APE→Std)")
    print("=" * 80)

    for max_iter in Config.LB_ITERATIONS:
        for alpha in Config.LB_ALPHA_VALUES:
            config_count += 1
            print(f"\n[Config {config_count}/{total_configs}] "
                  f"L-BFGS  α={alpha}  iter={max_iter}")
            test_generator.reset()
            lb_d2[max_iter][alpha] = evaluate_ape_to_std(
                model_apegan, model_standard, test_generator,
                attack_fn=lbfgs_attack_batch,
                attack_kwargs={'alpha': alpha, 'max_iter': max_iter},
                param_label=f"α={alpha} iter={max_iter}"
            )

    # ─────────────────────────────────────────────
    # SUMMARY PRINT
    # ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n── DeepFool ──")
    for max_iter in Config.DF_ITERATIONS:
        print(f"\n  iter={max_iter}")
        print(f"  {'δ':>5} | {'D1 Std WB':>10} | {'D1 APE Trf':>11} | "
              f"{'D1 Gap':>8} | {'D2 APE WB':>10} | {'D2 Std Trf':>11} | "
              f"{'D2 Gap':>8} | {'Asym':>6}")
        for ov in Config.DF_OVERSHOOT_VALUES:
            d1  = Config.EXP3_D1[max_iter][ov]
            d2  = df_d2[max_iter][ov]
            g1  = round(d1['std_wb'] - d1['ape_trf'], 2)
            g2  = d2['transferability_gap']
            asy = round(abs(g1 - g2), 2)
            print(f"  {ov:>5} | {d1['std_wb']:>9.2f}% | {d1['ape_trf']:>10.2f}% | "
                  f"{g1:>7.2f}% | {d2['apegan_whitebox_adv_accuracy']:>9.2f}% | "
                  f"{d2['standard_transfer_accuracy']:>10.2f}% | "
                  f"{g2:>7.2f}% | {asy:>5.2f}%")

    print("\n── L-BFGS ──")
    for max_iter in Config.LB_ITERATIONS:
        print(f"\n  iter={max_iter}")
        print(f"  {'α':>5} | {'D1 Std WB':>10} | {'D1 APE Trf':>11} | "
              f"{'D1 Gap':>8} | {'D2 APE WB':>10} | {'D2 Std Trf':>11} | "
              f"{'D2 Gap':>8} | {'Asym':>6}")
        for al in Config.LB_ALPHA_VALUES:
            d1  = Config.EXP4_D1[max_iter][al]
            d2  = lb_d2[max_iter][al]
            g1  = round(d1['std_wb'] - d1['ape_trf'], 2)
            g2  = d2['transferability_gap']
            asy = round(abs(g1 - g2), 2)
            print(f"  {al:>5} | {d1['std_wb']:>9.2f}% | {d1['ape_trf']:>10.2f}% | "
                  f"{g1:>7.2f}% | {d2['apegan_whitebox_adv_accuracy']:>9.2f}% | "
                  f"{d2['standard_transfer_accuracy']:>10.2f}% | "
                  f"{g2:>7.2f}% | {asy:>5.2f}%")

    # ─────────────────────────────────────────────
    # METADATA — clean, single-table ready
    # ─────────────────────────────────────────────
    evaluation_time = (datetime.now() - start_time).total_seconds()

    def build_df_block(max_iter):
        rows = {}
        gaps_d1, gaps_d2 = [], []
        for ov in Config.DF_OVERSHOOT_VALUES:
            d1  = Config.EXP3_D1[max_iter][ov]
            d2  = df_d2[max_iter][ov]
            g1  = round(d1['std_wb'] - d1['ape_trf'], 2)
            g2  = d2['transferability_gap']
            asy = round(abs(g1 - g2), 2)
            gaps_d1.append(g1)
            gaps_d2.append(g2)
            rows[str(ov)] = {
                'direction_1_std_to_apegan': {
                    'source': 'Experiment 3 Table II',
                    'std_whitebox_adv_accuracy': d1['std_wb'],
                    'apegan_transfer_accuracy':  d1['ape_trf'],
                    'transferability_gap':       g1
                },
                'direction_2_apegan_to_std': {
                    'source': 'Experiment 5 (new)',
                    'apegan_clean_accuracy':         d2['apegan_clean_accuracy'],
                    'apegan_whitebox_adv_accuracy':  d2['apegan_whitebox_adv_accuracy'],
                    'apegan_accuracy_drop':          d2['apegan_accuracy_drop'],
                    'standard_clean_accuracy':       d2['standard_clean_accuracy'],
                    'standard_transfer_accuracy':    d2['standard_transfer_accuracy'],
                    'standard_accuracy_drop':        d2['standard_accuracy_drop'],
                    'transferability_gap':           g2,
                    'total_samples':                 d2['total_samples']
                },
                'asymmetry': asy
            }
        mean_g1  = round(float(np.mean(gaps_d1)), 2)
        mean_g2  = round(float(np.mean(gaps_d2)), 2)
        mean_asy = round(abs(mean_g1 - mean_g2), 2)
        return {
            'per_param': rows,
            'mean': {
                'std_whitebox_adv_accuracy':    round(float(np.mean([Config.EXP3_D1[max_iter][ov]['std_wb']   for ov in Config.DF_OVERSHOOT_VALUES])), 2),
                'apegan_transfer_accuracy':     round(float(np.mean([Config.EXP3_D1[max_iter][ov]['ape_trf']  for ov in Config.DF_OVERSHOOT_VALUES])), 2),
                'mean_d1_gap':                  mean_g1,
                'apegan_whitebox_adv_accuracy': round(float(np.mean([df_d2[max_iter][ov]['apegan_whitebox_adv_accuracy'] for ov in Config.DF_OVERSHOOT_VALUES])), 2),
                'standard_transfer_accuracy':   round(float(np.mean([df_d2[max_iter][ov]['standard_transfer_accuracy']   for ov in Config.DF_OVERSHOOT_VALUES])), 2),
                'mean_d2_gap':                  mean_g2,
                'mean_asymmetry':               mean_asy
            }
        }

    def build_lb_block(max_iter):
        rows = {}
        gaps_d1, gaps_d2 = [], []
        for al in Config.LB_ALPHA_VALUES:
            d1  = Config.EXP4_D1[max_iter][al]
            d2  = lb_d2[max_iter][al]
            g1  = round(d1['std_wb'] - d1['ape_trf'], 2)
            g2  = d2['transferability_gap']
            asy = round(abs(g1 - g2), 2)
            gaps_d1.append(g1)
            gaps_d2.append(g2)
            rows[str(al)] = {
                'direction_1_std_to_apegan': {
                    'source': 'Experiment 4 Table III',
                    'std_whitebox_adv_accuracy': d1['std_wb'],
                    'apegan_transfer_accuracy':  d1['ape_trf'],
                    'transferability_gap':       g1
                },
                'direction_2_apegan_to_std': {
                    'source': 'Experiment 5 (new)',
                    'apegan_clean_accuracy':         d2['apegan_clean_accuracy'],
                    'apegan_whitebox_adv_accuracy':  d2['apegan_whitebox_adv_accuracy'],
                    'apegan_accuracy_drop':          d2['apegan_accuracy_drop'],
                    'standard_clean_accuracy':       d2['standard_clean_accuracy'],
                    'standard_transfer_accuracy':    d2['standard_transfer_accuracy'],
                    'standard_accuracy_drop':        d2['standard_accuracy_drop'],
                    'transferability_gap':           g2,
                    'total_samples':                 d2['total_samples']
                },
                'asymmetry': asy
            }
        mean_g1  = round(float(np.mean(gaps_d1)), 2)
        mean_g2  = round(float(np.mean(gaps_d2)), 2)
        mean_asy = round(abs(mean_g1 - mean_g2), 2)
        return {
            'per_param': rows,
            'mean': {
                'std_whitebox_adv_accuracy':    round(float(np.mean([Config.EXP4_D1[max_iter][al]['std_wb']   for al in Config.LB_ALPHA_VALUES])), 2),
                'apegan_transfer_accuracy':     round(float(np.mean([Config.EXP4_D1[max_iter][al]['ape_trf']  for al in Config.LB_ALPHA_VALUES])), 2),
                'mean_d1_gap':                  mean_g1,
                'apegan_whitebox_adv_accuracy': round(float(np.mean([lb_d2[max_iter][al]['apegan_whitebox_adv_accuracy'] for al in Config.LB_ALPHA_VALUES])), 2),
                'standard_transfer_accuracy':   round(float(np.mean([lb_d2[max_iter][al]['standard_transfer_accuracy']   for al in Config.LB_ALPHA_VALUES])), 2),
                'mean_d2_gap':                  mean_g2,
                'mean_asymmetry':               mean_asy
            }
        }

    metadata = {
        'experiment': {
            'name':                    'exp5_transferability',
            'description':             (
                'Bidirectional adversarial transferability analysis. '
                'DeepFool and L-BFGS attacks, each with 3 selected parameter values '
                'and 2 iteration configs. Direction 1 (Std→APE) from Exp 3/4 tables. '
                'Direction 2 (APE→Std) newly computed. '
                'All attack functions verbatim from Exp 3 / Exp 4.'
            ),
            'date':                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_time_seconds': evaluation_time,
            'evaluation_time_minutes': round(evaluation_time / 60, 2)
        },
        'attacks': {
            'deepfool': {
                'formula_paper': "x_adv = x - delta * (grad_f / ||grad_f||)",
                'formula_code':  "x[i] = x[i] - overshoot * grad[0] / grad_norm",
                'source':        'Experiment 3 verbatim',
                'overshoot_values': Config.DF_OVERSHOOT_VALUES,
                'iterations':       Config.DF_ITERATIONS
            },
            'lbfgs': {
                'formula_paper': "x_adv = x - alpha * grad_L(x, y)",
                'formula_code':  "x = x - alpha * grad.numpy()",
                'source':        'Experiment 4 verbatim',
                'alpha_values':  Config.LB_ALPHA_VALUES,
                'iterations':    Config.LB_ITERATIONS
            }
        },
        'dataset': {
            'name':         'EMNIST Balanced',
            'test_samples': test_generator.samples,
            'num_classes':  Config.NUM_CLASSES,
            'batch_size':   Config.BATCH_SIZE,
            'rescale':      '1./255',
            'target_size':  [28, 28],
            'color_mode':   'grayscale',
            'shuffle':      False
        },
        'baseline_clean_accuracy': {
            'standard_cnn': Config.STD_CLEAN_ACC,
            'apegan_cnn':   Config.APEGAN_CLEAN_ACC,
            'source':       'Paper Table I'
        },
        'column_definitions': {
            'std_whitebox_adv_accuracy':   'Standard CNN accuracy on adversarials crafted against itself (D1, white-box)',
            'apegan_transfer_accuracy':    'APE-GAN CNN accuracy on adversarials crafted against Standard CNN (D1, black-box)',
            'd1_gap':                      'std_wb - ape_trf  (negative = adversarials do not transfer to APE-GAN)',
            'apegan_whitebox_adv_accuracy':'APE-GAN CNN accuracy on adversarials crafted against itself (D2, white-box)',
            'standard_transfer_accuracy':  'Standard CNN accuracy on adversarials crafted against APE-GAN CNN (D2, black-box)',
            'd2_gap':                      'ape_wb - std_trf  (negative = adversarials do not transfer to Standard CNN)',
            'asymmetry':                   '|d1_gap - d2_gap|  (directional imbalance in transferability)'
        },
        'results': {
            'deepfool': {
                '5_iterations':  build_df_block(5),
                '10_iterations': build_df_block(10)
            },
            'lbfgs': {
                '40_iterations': build_lb_block(40),
                '80_iterations': build_lb_block(80)
            }
        }
    }

    metadata_path = os.path.join(Config.OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\n✓ Metadata saved: {metadata_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 5 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Evaluation Time : {evaluation_time / 60:.2f} minutes")
    print(f"Test Samples          : {test_generator.samples}")
    print(f"Configs Computed (D2) : {total_configs}")
    print(f"Metadata              : {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()