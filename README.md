# A Comparative Study of 5 Supervised Learning Algorithms

**Goal:** train five different algorithm *families* on the same dataset, under
identical conditions, to build intuition for which family suits which kind of
problem — rather than to squeeze out the last 0.1% of accuracy on any one model.

---

## 1. Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset** — UCI Machine Learning
Repository (bundled in scikit-learn as `load_breast_cancer`).

| | |
|---|---|
| Samples | 569 |
| Features | 30 numeric, real-valued (mean/SE/"worst" of 10 measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |
| Target | Binary classification — malignant (0) vs. benign (1) |
| Class balance | 212 malignant / 357 benign (~63% / 37%) — mildly imbalanced |
| Missing values | None |
| Categorical features | None (all continuous) |

This is a real, well-studied clinical dataset computed from digitized images
of fine needle aspirate (FNA) biopsies, and it has properties that make it a
good stress test for comparing algorithm families: many features (30),
several of which are **highly correlated** with each other (e.g. radius,
perimeter, and area are near-duplicates of the same underlying shape
information), and a **largely linearly-separable** structure once scaled.

## 2. Methodology (identical for every model)

1. **Missing values**: handled with a median `SimpleImputer` (defensive —
   this dataset happens to have none, but the pipeline is written so it
   generalizes to messier data).
2. **Categorical encoding**: `pd.get_dummies` step included in the pipeline
   for generality, though this dataset has no categorical columns to encode.
3. **Train/test split**: single stratified 80/20 split
   (`train_test_split(..., stratify=y, random_state=42)`), reused for every
   model — nobody gets an easier or harder test set.
4. **Scaling**: `StandardScaler` fit **only on the training set**, then
   applied to both train and test (no leakage). Logistic Regression, KNN,
   SVM, and Gaussian Naive Bayes are trained on the **scaled** features
   (they are distance/gradient/likelihood-based and are sensitive to feature
   magnitude). The Decision Tree is trained on the **raw, unscaled** features,
   since threshold-based splits are invariant to monotonic rescaling — scaling
   would not change its decision boundary at all.
5. **Models**, all from scikit-learn, all evaluated on the same held-out test
   set:
   - `LogisticRegression(max_iter=5000)`
   - `DecisionTreeClassifier(max_depth=5)`
   - `KNeighborsClassifier(n_neighbors=7)`
   - `SVC(kernel="rbf", probability=True)`
   - `GaussianNB()`
6. **Metrics**: Accuracy, Precision, Recall, F1-score, and ROC-AUC — all
   appropriate for binary classification, and reported together since
   accuracy alone can be misleading on an imbalanced target.

## 3. Results

Ranked by F1-score (test set, n=114):

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Input Features |
|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | 0.9825 | 0.9861 | 0.9861 | **0.9861** | 0.9954 | Scaled |
| 2 | **Support Vector Machine (RBF)** | 0.9825 | 0.9861 | 0.9861 | **0.9861** | 0.9950 | Scaled |
| 3 | K-Nearest Neighbors | 0.9737 | 0.9600 | 1.0000 | 0.9796 | 0.9884 | Scaled |
| 4 | Gaussian Naive Bayes | 0.9298 | 0.9444 | 0.9444 | 0.9444 | 0.9868 | Scaled |
| 5 | Decision Tree | 0.9211 | 0.9565 | 0.9167 | 0.9362 | 0.9163 | Raw (unscaled) |

![Ranking chart](results/ranking_chart.png)

![Full metric comparison](results/comparison_chart.png)

(Raw numbers: `results/comparison_table.csv`. Per-model confusion matrices:
`results/confusion_matrices.json`.)

## 4. Analysis: why the best model won, and why the worst struggled

### 🥇 Winner: Logistic Regression (tied with SVM)

Logistic Regression came out on top (tied with the RBF-SVM) with an F1-score
of 0.986 and the highest ROC-AUC (0.995). This makes sense given the
structure of the data:

- **The classes are close to linearly separable.** Malignant vs. benign
  tumors differ mainly in *magnitude* — larger radius, higher concavity,
  coarser texture — which a linear decision boundary over standardized
  features can capture very well. When a simple linear model matches the
  true shape of the decision boundary, it wins, because it also has far
  lower variance than more flexible models.
- **High-dimensional but well-scaled continuous features are exactly
  Logistic Regression's comfort zone.** With 30 features and only ~455
  training rows, a linear model's small hypothesis space acts as a natural
  regularizer against overfitting, whereas more flexible non-linear models
  have more room to fit noise.
- **The RBF-SVM tying it** reinforces this reading: SVM with an RBF kernel
  *can* learn curved boundaries, but here it apparently converges to a
  boundary very close to linear (or a very gentle curve) — i.e., the extra
  flexibility wasn't needed, and the model found essentially the same
  solution as the linear one. When the true boundary is (near-)linear,
  linear models and margin-based models both do well; the linear model wins
  on simplicity and interpretability, since it gets there with far less
  computation and no kernel/hyperparameter search.

### 🥈 Strong second tier: K-Nearest Neighbors

KNN (k=7, scaled features) reached F1 = 0.980, with **perfect recall**
(1.0) but lower precision (0.96) — it correctly caught every malignant
case in the test set but had a few more false positives. This tracks with
how KNN works: once features are standardized, "malignant" tumors cluster
together in feature space because the underlying measurements (radius,
concavity, etc.) genuinely covary with diagnosis. KNN exploits that local
structure directly without needing to assume a global linear or Gaussian
form, which is why it does almost as well as Logistic Regression here even
though it makes no assumptions about the decision boundary's shape at all.

### 🥉 Middling: Gaussian Naive Bayes

Naive Bayes dropped to F1 = 0.944 — noticeably behind the top three. The
data explains why:

- **Its core assumption is violated.** Naive Bayes assumes features are
  *conditionally independent* given the class. But in this dataset, radius,
  perimeter, and area are essentially the same geometric quantity measured
  three different ways (and this repeats across "mean", "standard error",
  and "worst" versions of each of 10 base measurements) — the average
  pairwise absolute correlation across the 30 features is **~0.40**, and
  nearly 5% of feature pairs are correlated above 0.8. Naive Bayes
  effectively "double counts" this redundant evidence, which skews its
  posterior probabilities and hurts precision/recall relative to models
  that can natively handle correlated inputs.
- **The Gaussian-per-feature assumption is also only approximate** — several
  of these measurements are right-skewed rather than normally distributed,
  which further weakens the fit of the class-conditional likelihoods.
- Its still-high ROC-AUC (0.987) shows the model *ranks* cases sensibly —
  it's the miscalibrated decision threshold from the independence
  violation, not a lack of signal, that costs it accuracy at the default
  0.5 cutoff.

### 🚫 Weakest: Decision Tree

The single Decision Tree (max depth 5) came in last, at F1 = 0.936, and
notably had by far the lowest ROC-AUC (0.916) of the five models. A few
data-driven reasons:

- **A single tree is high-variance and this dataset is small.** With only
  ~455 training rows and 30 continuous features, small differences in the
  training split can lead the tree to pick a slightly different first split,
  which cascades into a very different tree structure. Ensembles (Random
  Forest, Gradient Boosting) fix this by averaging over many trees, but a
  lone tree has no such stabilizer — this is exactly the well-known
  weakness of decision trees on small-to-medium tabular data.
- **Axis-aligned splits are a poor match for correlated, continuous
  features.** Because a tree can only split on one feature at a time with a
  single threshold, it struggles to efficiently represent a decision
  boundary that really depends on a *combination* of several correlated
  measurements (e.g. "radius AND concavity together"). Linear/margin
  models, and even distance-based KNN, capture that combined effect much
  more naturally.
- **Depth limiting (max_depth=5) trades variance for bias.** Restricting
  depth was necessary to avoid even worse overfitting, but it also caps how
  much of the feature interaction structure the tree can represent,
  contributing to its comparatively low ROC-AUC — it's not just getting
  individual cases wrong, it's ranking uncertain cases less reliably than
  the other four models.

## 5. Takeaways — which family suits which kind of problem

| Situation in your data | Algorithm family that tends to shine |
|---|---|
| Roughly linear / near-separable classes, moderate-to-high dimensionality, limited data | **Linear models** (Logistic/Linear Regression) — low variance, hard to beat when the true boundary really is close to linear |
| Boundary shape unknown, moderate dataset size, willing to tune a kernel | **SVM** — flexible margin-based boundary, but converges to near-linear behavior gracefully when that's what the data needs |
| Local neighborhoods in feature space are informative, no assumption about boundary shape wanted | **KNN** — simple and effective once features are scaled, but slower at prediction time and sensitive to irrelevant features |
| Features are cheap/fast probabilistic signals, low-latency need, and features are close to independent | **Naive Bayes** — very fast and works well *if* the independence assumption roughly holds; degrades as features become more correlated/redundant |
| Need interpretability, mixed feature types, non-linear interactions, but ideally as an ensemble (Random Forest/GBM) rather than a single tree | **Decision Trees** — the best entry point into ensemble methods, but a lone tree is high-variance and easily beaten by simpler linear models on small, clean, mostly-linear data like this one |

The core intuition this experiment reinforces: **algorithm choice should be
driven by the shape of your decision boundary, your feature correlation
structure, and your sample size — not by which model is "fanciest."** Here,
the simplest model (Logistic Regression) tied for first place precisely
because the data's true structure was close to linear, and the most
seemingly "sensible for tabular data" choice (a single Decision Tree) came
in last because it's poorly suited to a small sample of correlated
continuous features.

## 6. Reproducing this study

```bash
pip install scikit-learn pandas matplotlib seaborn
python comparative_study.py
```

Outputs are written to `results/`:
- `comparison_table.csv` — the ranked metrics table
- `comparison_chart.png` — grouped bar chart of all 5 metrics × all 5 models
- `ranking_chart.png` — headline F1-score ranking
- `confusion_matrices.json` — per-model confusion matrices

## 7. Files in this project

```
ml_comparative_study/
├── comparative_study.py      # full pipeline: load → preprocess → train → evaluate → chart
├── README.md                  # this file
└── results/
    ├── comparison_table.csv
    ├── comparison_chart.png
    ├── ranking_chart.png
    └── confusion_matrices.json
```
