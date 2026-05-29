# Adversarial Robustness of CNN on EMNIST Balanced

> 📬 Paper submitted to SN Computer Science (Springer Nature), 2026.

## What This Project Does

Trained and evaluated a custom CNN against adversarial attacks on the EMNIST
Balanced dataset (47-class alphanumeric recognition, 131,600 images). Implemented
APE-GAN adversarial training as a defense and benchmarked it against two attacks
across multiple configurations. Conducted a full bidirectional transferability analysis
to assess cross-model black-box robustness.

---

## Key Results

### Performance on Clean Test Data

| Evaluation | Standard CNN | APE-GAN CNN | Improvement |
|---|---|---|---|
| Clean Data | 90.18% | 88.66% | −1.52% |

### Performance Comparison: Standard vs APE-GAN CNN

| Evaluation | Standard | APE-GAN | Improvement |
|---|---|---|---|
| Clean Data | 90.18% | 88.66% | −1.52% |
| DeepFool — 5 iterations (mean) | 43.03% | 84.44% | **+41.41%** |
| DeepFool — 10 iterations (mean) | 19.05% | 77.90% | **+58.85%** |
| L-BFGS — 40 iterations (mean) | 64.91% | 85.18% | **+20.27%** |
| L-BFGS — 80 iterations (mean) | 58.16% | 84.28% | **+26.12%** |

### Adversarial Transferability: Accuracy (%) on Test Set

| Param | Std WB | APE Trf | Gap D1 | APE WB | Std Trf | Gap D2 | Asym (%) |
|---|---|---|---|---|---|---|---|
| **DeepFool — 5 Iterations** | | | | | | | |
| δ=0.1 | 81.77 | 88.12 | −6.35 | 79.23 | 89.81 | −10.58 | 4.23 |
| δ=0.3 | 46.06 | 85.06 | −39.00 | 56.18 | 88.41 | −32.23 | 6.77 |
| δ=0.6 | 14.36 | 80.68 | −66.32 | 29.99 | 84.72 | −54.73 | 11.59 |
| **Mean** | **47.40** | **84.62** | **−37.22** | **55.13** | **87.65** | **−32.51** | **4.71** |
| **DeepFool — 10 Iterations** | | | | | | | |
| δ=0.1 | 61.13 | 86.71 | −25.58 | 65.31 | 89.29 | −23.98 | 1.60 |
| δ=0.3 | 11.16 | 80.51 | −69.35 | 25.79 | 84.74 | −58.95 | 10.40 |
| δ=0.6 | 4.84 | 67.44 | −62.60 | 8.60 | 70.87 | −62.27 | 0.33 |
| **Mean** | **25.71** | **78.22** | **−52.51** | **33.23** | **81.63** | **−48.40** | **4.11** |
| **L-BFGS — 40 Iterations** | | | | | | | |
| α=1.0 | 74.41 | 86.79 | −12.38 | 68.31 | 88.53 | −20.22 | 7.84 |
| α=3.0 | 65.11 | 85.32 | −20.21 | 57.48 | 86.87 | −29.39 | 9.18 |
| α=6.0 | 58.58 | 83.93 | −25.35 | 50.32 | 84.77 | −34.45 | 9.10 |
| **Mean** | **66.03** | **85.35** | **−19.31** | **58.70** | **86.72** | **−28.02** | **8.71** |
| **L-BFGS — 80 Iterations** | | | | | | | |
| α=1.0 | 68.40 | 85.90 | −17.50 | 61.23 | 87.65 | −26.42 | 8.92 |
| α=3.0 | 58.18 | 84.40 | −26.22 | 49.78 | 85.48 | −35.70 | 9.48 |
| α=6.0 | 51.37 | 83.07 | −31.70 | 42.80 | 83.29 | −40.49 | 8.79 |
| **Mean** | **59.32** | **84.46** | **−25.14** | **51.27** | **85.47** | **−34.20** | **9.06** |

Std = Standard CNN. APE = APE-GAN CNN. WB = white-box adversarial accuracy on source model.
Trf = transfer accuracy on target model. Gap = WB − Trf. Asym = |Gap D1 − Gap D2|.
Direction 1 (Std→APE): adversarials crafted against Standard CNN. Direction 2 (APE→Std): adversarials crafted against APE-GAN CNN.

---

## What I Built

- Diagnosed severe overfitting (14.4% train-val gap) in the reference architecture
  and redesigned the CNN with progressive filter scaling (32→64→128), batch
  normalization, dropout, L2 regularization and data augmentation — reducing the
  gap to ~2% while improving test accuracy from 85.5% to 90.18%
- Implemented APE-GAN adversarial training from scratch: mixed batches of 50%
  clean and 50% FGSM-generated adversarial samples (ε=0.3) per epoch
- Evaluated both models against DeepFool (6 overshoot values × 2 iteration counts)
  and L-BFGS (6 step sizes × 2 iteration counts) on the full 18,800-image test set
- Ran a bidirectional adversarial transferability analysis (Std→APE and APE→Std)
  confirming both models resist black-box attacks across all evaluated configurations

## Stack

Python · TensorFlow / Keras · NumPy · Matplotlib · Seaborn · Kaggle (T4 GPU × 2)

---

## Scripts

| File | What it does |
|---|---|
| `baseline_cnn_reference.py` | Reproduces reference architecture, exposes overfitting |
| `standard_cnn_training.py` | Trains proposed CNN — 90.18% test accuracy |
| `apegan_adversarial_training.py` | APE-GAN adversarial training — 88.66% test accuracy |
| `deepfool_attack_evaluation.py` | DeepFool attack evaluation across 6 overshoot values × 2 iteration counts |
| `lbfgs_attack_evaluation.py` | L-BFGS attack evaluation across 6 step sizes × 2 iteration counts |
| `transferability_analysis.py` | Bidirectional black-box transferability analysis (Std→APE and APE→Std) |

---

## Pre-trained Models

Pre-trained models are not included in this repository.
To reproduce results, run the training scripts in order:

1. `standard_cnn_training.py`
2. `apegan_adversarial_training.py`

Then run the evaluation scripts against the saved models.

---

## License

MIT