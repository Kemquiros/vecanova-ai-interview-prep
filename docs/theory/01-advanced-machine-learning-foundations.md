# Theory 1: Advanced Machine Learning Foundations for AI Engineers

The classical-ML half of the senior loop: metric selection under business constraints,
gradient-boosting internals, anomaly detection, recommendation systems, calibration,
explainability, and drift. The framing is deliberately opinionated: interviewers at
FAANG-tier companies and global consulting firms are not checking whether you can recite
the F1 formula. They are checking whether you can defend a metric choice against a cost
model, state when you would change your mind, and connect the math to a shipped system.

## Interview Soundbites

- "ROC-AUC only tells me the model ranks well. In fraud I pick the threshold from the
  review-capacity budget — how many cases the ops team can clear per day — and report
  precision at that operating point, because that is the number the business feels."
- "I treat calibration as a first-class deliverable, not a bonus: a model with 0.92 AUC
  and miscalibrated probabilities still breaks every downstream cost calculation that
  multiplies probability by impact."
- "Gradient boosting is functional gradient descent in function space: each tree fits the
  negative gradient of the loss, and XGBoost's contribution is a second-order Taylor
  expansion of that loss plus explicit regularization, which makes leaf weights
  closed-form."
- "Isolation Forest works because anomalies are 'few and different': a random axis-parallel
  split isolates them in fewer cuts, so expected path length converts directly into an
  anomaly score between 0 and 1."
- "For drift I run PSI with the 0.1 / 0.25 thresholds plus a KS test on the model score,
  because coarse PSI bins can hide within-bin shape changes that actually move the
  decision boundary."

## 1. Classification Metrics: Choose the Metric Before the Model

The metric is a business decision, not a modeling detail. State the asymmetry first, then
the formula.

$$\text{Precision} = \frac{TP}{TP + FP} \qquad
\text{Recall} = \frac{TP}{TP + FN} \qquad
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Precision** dominates when false positives are expensive in operations, not just in
  trust: flagging a legitimate transaction in fintech costs a human review slot and a
  customer; alerting on a healthy device in a SaaS observability product pages an on-call
  engineer.
- **Recall** dominates when false negatives are catastrophic: a missed sepsis flag in
  healthcare or a missed money-laundering pattern in KYC screening carries regulatory and
  human cost that dwarfs alert fatigue.
- **F1** is the default only when you genuinely cannot rank the two errors; say out loud
  that it weights precision and recall equally, and that a business-weighted
  $F_\beta$ (with $\beta > 1$ favoring recall) is the honest version when the client
  cares more about misses.

For imbalanced problems, argue in the PR plane, not the ROC plane. ROC-AUC is computed
over TPR and FPR; FPR's denominator is the enormous negative class, so with 0.5 percent
positives a model can look great on ROC while producing unusable precision at every
plausible threshold. PR-AUC's baseline is the prevalence itself, so gains are visible and
honest.

| Dimension | ROC-AUC | PR-AUC |
| --- | --- | --- |
| Plane | TPR vs FPR | Precision vs Recall |
| Class imbalance | Overly optimistic: FPR denominator is the huge negative class | Realistic: precision degrades as positives thin out |
| Random-classifier baseline | 0.5 regardless of prevalence | Equal to positive prevalence |
| Best for | Balanced classes, pure ranking quality | Rare positives: fraud under 1 percent, sepsis under 5 percent |
| Blind spot | Says nothing about the operating point | Ignores the recall side of the full ranking |

Two senior differentiators. First, **MCC** (Matthews Correlation Coefficient) uses all
four confusion cells and stays informative under imbalance:

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Second, **calibration**, which no ranking metric measures. A model can have superb AUC
while its "0.8" events happen 55 percent of the time. Measure it with the Brier score and
Expected Calibration Error:

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2 \qquad
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\lvert \text{acc}(b) - \text{conf}(b) \rvert$$

Fix calibration with Platt scaling (small data, parametric sigmoid) or isotonic regression
(thousands of samples, non-parametric), or temperature scaling for neural networks — always
fit the recalibrator on a holdout, never on training data. Healthcare risk scores are the
canonical case: a clinician acting on "8 percent probability" consumes the calibration,
not the AUC.

## 2. Regression Metrics and Their Failure Modes

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} \lvert y_i - \hat{y}_i \rvert \qquad
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

- MAE minimizes the median (L1); RMSE minimizes the mean (L2). RMSE punishes large errors
  quadratically, so it is the right objective when a big miss is disproportionately bad: a
  logistics network that promises 3 days and takes 9 is worse than twice as bad as one
  that takes 6.
- MAE is the right reporting metric when outliers are noise you refuse to let dominate the
  KPI. State the pairing explicitly: optimize with one, report with both, and add a
  quantile metric (Pinball Loss) when the product needs an upper bound rather than a point
  forecast:

$$L_\tau(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}) & y \ge \hat{y} \\ (1-\tau)(\hat{y} - y) & y < \hat{y} \end{cases}$$

For demand forecasting, demand-censored SKUs (stockouts hide true demand) make naive RMSE
misleading: train on unbiased demand estimates or predict the 90th-percentile quantile and
measure service level directly.

## 3. Gradient Boosting Internals

Friedman's stagewise additive model: at stage $m$, fit a weak learner $h_m$ to the
negative gradient of the loss evaluated at the current ensemble, then take a shrunken step:

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x), \qquad
h_m \approx -\frac{\partial L(y, F(x))}{\partial F(x)}\Big|_{F = F_{m-1}}$$

where $\nu$ is the learning rate (shrinkage). XGBoost's refinement: Taylor-expand the loss
to second order around $F_{m-1}$, so each leaf carries a gradient sum $G$ and a Hessian
sum $H$, and the optimal leaf weight and split gain are closed-form:

$$w^* = -\frac{G}{H + \lambda}$$

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda}
- \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

with the objective regularized as $\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2$
($T$ leaf count, $\gamma$ split penalty, $\lambda$ L2 on weights). Deriving this on a
whiteboard separates senior from mid-level: $\gamma$ prunes splits whose gain is negative,
$\lambda$ shrinks leaf weights, and row/column subsampling adds stochastic regularization
on top.

| Dimension | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| Split finding | Pre-sorted / quantile sketch | Histogram-based, low-memory | Oblivious (symmetric) trees |
| Growth strategy | Level-wise | Leaf-wise (loss-driven) | Level-wise, symmetric depth |
| Categorical features | One-hot / target encoding upstream | Native via gradient statistics | Native, ordered target statistics (leak-resistant) |
| Speed profile | Baseline | Fastest at millions of rows | Slower; strongest on categorical-heavy data |
| Overfitting risk | Moderate, needs tuning | Higher with leaf-wise growth; cap `num_leaves` | Lower: symmetric trees regularize structurally |
| Pick when | General tabular, widest ecosystem | Large-scale tabular, fast retraining | Many high-cardinality categoricals, strict leakage concerns |

Missing values are handled natively in XGBoost by learning a default split direction per
node — say this precisely, because "imputes automatically" is wrong. Monotonic constraints
are available in all three and are a genuine differentiator in credit and clinical scoring,
where a regulator will ask why the score fell as income rose.

## 4. Anomaly Detection

**Isolation Forest** builds random axis-parallel trees. The expected path length to isolate
a point, $E[h(x)]$, is short for anomalies because they sit in sparse regions; the score
normalizes it against the average path length $c(n)$ of an unsuccessful search in a binary
search tree of $n$ samples ($H(i) \approx \ln(i) + 0.5772$, the Euler-Mascheroni constant):

$$s(x, n) = 2^{-E[h(x)] / c(n)}, \qquad c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

$s \to 1$ as $E[h(x)] \to 0$ (anomalous), $s \to 0.5$ for inliers. No distance metric, no
density estimate, sublinear at scoring time — which is why it is the default tripwire for
transaction streams and device telemetry.

**Autoencoders** learn a low-dimensional reconstruction of normal behavior;
reconstruction error $\lVert x - \hat{x} \rVert^2$ flags points the manifold cannot
represent. They shine on structured signals (sensor windows, images) but fail quietly: if
anomalies are more than a small fraction of training data, the model learns to reconstruct
them too, and your alerting dies without an error.

| Dimension | Isolation Forest | Autoencoder | Distributional (PSI/KS on scores) |
| --- | --- | --- | --- |
| Assumption | Anomalies isolate in fewer random splits | Normal data lies on a learnable manifold | Output distribution is stationary |
| Data requirement | Unlabeled, works on modest sample sizes | Large corpus of normal samples | Reference window of scores |
| Interpretability | Path length per point | Per-feature reconstruction error | Threshold distance, easy to chart |
| Streaming cost | $O(\log n)$ per point | One forward pass | $O(1)$ binning |
| Signature failure | Local anomalies inside dense clusters score low | Learns to reconstruct anomalies if contaminated | Coarse bins hide within-bin drift |
| Pick when | First baseline on tabular streams | Rich structured signals with exploitable structure | Guardrail layered over any detector |

## 5. Recommendation Systems and Embeddings

Classic matrix factorization learns user and item latent factors jointly; alternating
least squares (ALS) solves each side in closed form while holding the other fixed:

$$\min_{X, Y} \sum_{(u,i) \in K} \left( r_{ui} - x_u^\top y_i \right)^2
+ \lambda \left( \lVert x_u \rVert^2 + \lVert y_i \rVert^2 \right)$$

Collaborative filtering exploits behavior but collapses on cold start; content-based
filtering exploits item metadata and handles new items but caps discovery (it only ever
recommends more of what the user already liked). Production recommenders are two-stage: a
fast candidate generator (ANN over embeddings, or ALS) proposes hundreds of items, then a
heavier ranker (gradient boosting on features, or a cross-encoder) orders them — the same
recall/rerank split as modern RAG, and worth naming in the interview.

Cosine similarity is the shared primitive across recommenders and semantic search:

$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert}$$

Watch for popularity bias offline (a naive NDCG@k can be maximized by recommending
blockbusters) and treat offline metrics as a filter, not an oracle: recall@k and NDCG@k to
gate candidates, then an online A/B with guardrail metrics (session diversity, long-term
retention) to confirm.

## 6. Drift: PSI and KS as the Standing Guardrail

Data drift, not model decay, is the dominant cause of silent degradation in deployed
classical models. The Population Stability Index over $B$ bins, with $e_i$ the reference
proportion and $a_i$ the live proportion in bin $i$:

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i) \ln \frac{a_i}{e_i}$$

Standard operating thresholds, and the exact contract implemented in
`src/interview_prep/drift.py`: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 severe.
The two-sample Kolmogorov-Smirnov statistic is the complementary check:

$$D_{n,m} = \sup_x \lvert F_{1,n}(x) - F_{2,m}(x) \rvert$$

PSI is bin-resolution-dependent and can hide a concentrated shift inside one coarse bin;
KS is sensitive to the largest CDF gap anywhere in the support. Run both, on the inputs
that matter and on the model score itself, and pair drift alerts with calibration
monitoring — a shifted population usually re-calibrates the score distribution before it
moves AUC.

## 7. Explanations: SHAP Values

SHAP assigns each feature $i$ an additive contribution $\phi_i$ by averaging its marginal
contribution over all feature coalitions $S \subseteq F \setminus \{i\}$:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}}
\frac{\lvert S \rvert! \, (\lvert F \rvert - \lvert S \rvert - 1)!}{\lvert F \rvert!}
\left( v(S \cup \{i\}) - v(S) \right)$$

The properties that matter to a regulator are efficiency (contributions plus the base
value sum exactly to the prediction) and consistency (if a feature's marginal effect grows,
its attribution cannot shrink). In practice: TreeSHAP for exact polynomial-time values on
tree ensembles; be honest that correlated features split credit arbitrarily, and pair
local SHAP with global summaries and counterfactual examples.

## Repository Cross-Reference

The concepts above are implemented in `src/interview_prep/traditional_ml.py` and
`src/interview_prep/drift.py`:

```python
from interview_prep.traditional_ml import MockAnomalyDetector, MockRecommenderSystem

detector = MockAnomalyDetector(threshold=2.5)
score = detector.compute_anomaly_score([38.1, 36.9, 37.2, 41.5])  # z-score proxy
flagged = detector.is_anomaly([38.1, 36.9, 37.2, 44.0])           # score > threshold

recommender = MockRecommenderSystem(
    item_embeddings={"care-plan-A": [0.9, 0.1], "care-plan-B": [0.2, 0.8]}
)
top = recommender.recommend([0.8, 0.2], top_k=2)  # cosine-similarity ranking
```

```python
from interview_prep.drift import drift_verdict, ks_statistic, population_stability_index

psi = population_stability_index(expected_scores, live_scores)
if drift_verdict(psi) != "STABLE":          # 0.1 / 0.25 thresholds
    ks = ks_statistic(expected_scores, live_scores)  # localize the shift
```

## Production War Stories / Pitfalls

- **Threshold inherited from ROC analysis.** A fraud model shipped with the default 0.5
  cutoff; the review queue received 40 times the team's daily capacity. Root cause: the
  metric (AUC) was chosen with no operating-point analysis. Mitigation: pick the threshold
  from the PR curve under a capacity constraint, monitor precision at that threshold, and
  re-tune it on every retrain.
- **Leakage through event-time joins.** A readmission model scored 0.94 AUC offline and
  0.61 online. Root cause: one feature (post-discharge medication fills) was only known
  after the prediction time. Mitigation: point-in-time joins with feature-store event-time
  semantics, and a leak scan that recomputes every feature as of the label timestamp.
- **Random splits on time series.** A demand model trained on 12 months including December
  and tested on January looked brilliant and failed at launch. Root cause: seasonality plus
  non-stationarity broke the i.i.d. assumption. Mitigation: rolling-origin (walk-forward)
  evaluation and per-horizon MAE/RMSE reporting.
- **SMOTE applied before the split.** Synthetic neighbors of test-set patients leaked into
  training, inflating recall by 15 points. Root cause: resampling outside the training
  fold. Mitigation: resample inside a pipeline fit per fold, or use class weights, which
  cannot leak.
- **PSI flat while the model decayed.** Five coarse quantile bins hid a shift concentrated
  in the top decile of the score distribution. Root cause: bin resolution. Mitigation:
  10-20 bins on the score, KS as a second opinion, and segment-level PSI for the segments
  the business actually acts on.
- **Threshold drift after champion-challenger swap.** The new champion was calibrated
  differently; the unchanged 0.35 cutoff collapsed recall. Root cause: thresholds treated
  as constants instead of model artifacts. Mitigation: store the threshold with the model
  artifact and gate releases on calibration metrics in CI.

## What Interviewers Actually Ask

- **"Your fraud model has 0.97 ROC-AUC but the client calls it useless. Diagnose."**
  - Ask for prevalence and the operating point before touching the model.
  - Likely causes: miscalibration, threshold set arbitrarily, precision unusable at the
    review capacity, or evaluation leakage inflating AUC.
  - Deliverable: PR curve, precision at the capacity-feasible threshold, calibration plot.
- **"How does XGBoost choose a split?"**
  - Second-order Taylor expansion of the loss; closed-form leaf weight $-G/(H+\lambda)$.
  - Gain formula compares left, right, and parent; approximate quantile sketches for
    scale; sparsity-aware default direction for missing values.
- **"When would you not use gradient boosting?"**
  - Raw text, images, audio (representation learning needed); streaming models requiring
    cheap incremental updates; ultra-low-latency with thousands of features (consider
    distilled linear models); strict extrapolation needs (trees cannot extrapolate).
  - Counterpoint to volunteer: monotonic constraints and interpretability via TreeSHAP
    make GBMs acceptable even in regulated scoring.
- **"Isolation Forest versus autoencoder for telemetry anomalies?"**
  - Few-and-different assumption vs manifold assumption; sample efficiency; streaming cost.
  - Practical answer: Isolation Forest as the tripwire, autoencoder for per-sensor
    attribution, distributional checks (PSI/KS) as the meta-guardrail.
- **"How do you evaluate a recommender offline?"**
  - Two-stage metrics: recall@k for candidates, NDCG@k for ranking; leave-last-out splits,
    never random splits, on temporal data.
  - Guard against popularity bias; offline is a filter, online A/B decides.
- **"The model is miscalibrated. Fix it."**
  - Measure ECE/Brier per segment; Platt for small data, isotonic for larger, temperature
    scaling for neural nets; refit on holdout; re-check after every retrain.
- **"Explain a single prediction to a regulator."**
  - SHAP with a documented baseline; efficiency property guarantees the decomposition adds
    up; state correlated-features caveat and pair with counterfactuals.

## Further Reading

- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (Annals of
  Statistics, 2001) — https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full
- Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System" (KDD 2016) —
  https://arxiv.org/abs/1603.02754
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017)
  — https://arxiv.org/abs/1711.08789
- Prokhorenkova et al., "CatBoost: Unbiased Boosting with Categorical Features" (NeurIPS
  2018) — https://arxiv.org/abs/1706.09516
- Liu, Ting, and Zhou, "Isolation Forest" (ICDM 2008) —
  https://www.cs.uic.edu/~zhangxz/publ/papers/isof.pdf
- Saito and Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot
  When Evaluating Binary Classifiers on Imbalanced Datasets" (PLOS ONE, 2015) —
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432
- Guo, Pleiss, Sun, and Weinberger, "On Calibration of Modern Neural Networks" (ICML 2017)
  — https://arxiv.org/abs/1706.04599
- Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
  — https://arxiv.org/abs/1705.07874
- Koren, Bell, and Volinsky, "Matrix Factorization Techniques for Recommender Systems"
  (IEEE Computer, 2009) — https://ieeexplore.ieee.org/document/5197422
