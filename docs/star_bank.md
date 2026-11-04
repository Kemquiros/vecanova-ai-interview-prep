# STAR Interview Question Bank (English Mastery)

## 1. Tell me about a time you handled a hallucination or failure in a production GenAI system.
- **Situation**: While deploying an enterprise RAG assistant for clinical protocols, initial user tests showed sporadic hallucination of medical dosages.
- **Task**: Eliminate hallucinations and establish rigorous evaluation gates without increasing p99 latency beyond 800ms.
- **Action**: I introduced a hybrid retrieval pipeline with a Cohere reranker and implemented RAGAS Faithfulness evaluation checks as part of our automated CI/CD pipeline using Langfuse.
- **Result**: Reduced hallucination rates by 85% and achieved 99.4% faithfulness across all production responses.

## 2. How do you handle PII and data privacy when integrating external LLM APIs?
- **Situation**: Working with a global healthcare provider subject to strict HIPAA and GDPR regulations.
- **Task**: Ensure zero PII leakage while leveraging state-of-the-art foundation models.
- **Action**: Built a pre-pipeline anonymization layer using Microsoft Presidio and regex to mask names, SSNs, and emails before payload transmission, combined with private VPC deployments on AWS SageMaker.
- **Result**: Achieved 100% regulatory compliance audit clearance with zero data leakage incidents.
