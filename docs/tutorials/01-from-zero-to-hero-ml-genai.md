# Tutorial 1: Bridging Traditional ML and GenAI

## 1. Objective
Understand how classical predictive machine learning (classification, regression, anomaly detection, recommendation systems) integrates directly into modern generative AI pipelines and multi-agent systems.

## 2. Core Concepts
- **Tabular Features as Context**: Generative LLMs excel at reasoning, but structured numerical data (blood pressure, anomaly scores, prediction probabilities) should be calculated via classical ML models (`XGBoost`, `Isolation Forests`) and injected as structured JSON context into prompts.
- **Anomaly Detection in Healthcare**: Using statistical deviation or Isolation Forests to flag patient telemetry anomalies before an agent triggers clinical alerts.

## 3. Code Walkthrough (`src/interview_prep/traditional_ml.py`)
```python
detector = MockAnomalyDetector(threshold=2.0)
score = detector.compute_anomaly_score([120.0, 122.0, 210.0, 119.0])
is_anom = detector.is_anomaly([120.0, 122.0, 210.0, 119.0])
```

## 4. Interview Narrative
"In my previous roles, I bridged traditional predictive modeling with GenAI by running tabular inference with XGBoost and Isolation Forests in backend services, then passing those structured outputs as deterministic context to LangGraph agents making clinical decisions."
