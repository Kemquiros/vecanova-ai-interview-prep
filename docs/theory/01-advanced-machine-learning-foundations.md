# Theory 1: Advanced Machine Learning Foundations for AI Engineers

## 1. Evaluation Metrics for Classification and Regression
When interviewing for Senior ML/AI positions, understanding when to use specific metrics is critical.
- **Classification**:
  - **Precision**: $TP / (TP + FP)$ — Crucial when False Positives are costly (e.g., flagging healthy tissue as cancerous).
  - **Recall**: $TP / (TP + FN)$ — Crucial when False Negatives are catastrophic (e.g., missing a disease diagnosis).
  - **F1-Score**: Harmonic mean of Precision and Recall: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$.
  - **ROC-AUC & PR-AUC**: Area under the ROC curve (True Positive Rate vs. False Positive Rate) and Precision-Recall curve. PR-AUC is preferred in highly imbalanced datasets (e.g., rare anomaly detection).
- **Regression**:
  - **MAE (Mean Absolute Error)**: Robust to outliers.
  - **RMSE (Root Mean Squared Error)**: Penalizes large errors heavily due to squaring.

## 2. Tree-Based Models & Gradient Boosting (`XGBoost`, `LightGBM`)
- How gradient boosting works: Iteratively training decision trees on the pseudo-residuals of the loss function.
- Regularization (`L1`/`L2` leaf weights, shrinkage/learning rate) to prevent overfitting.
- Handling missing values natively in XGBoost by learning default split directions.

## 3. Anomaly Detection (`Isolation Forests`, `Autoencoders`)
- **Isolation Forests**: Randomly selecting features and split values. Anomalies require fewer splits to isolate because they are "few and different" (shorter path length in isolation trees).
- **Autoencoders**: Neural network compression models trained to reconstruct normal data. High reconstruction error indicates an anomaly (fraud, clinical telemetry failure).

## 4. Recommendation Systems & Vector Embeddings
- **Collaborative Filtering**: Matrix factorization (SVD, Alternating Least Squares) based on user-item interaction matrices.
- **Content-Based Filtering**: Recommending items similar to those a user liked based on item metadata features.
- **Vector Embeddings & Semantic Search**: Representing items and queries as dense vectors in high-dimensional space, measured via cosine similarity:
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
