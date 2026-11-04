# Tutorial 3: AI Guardrails, PII, and Healthcare Compliance

## 1. Objective
Ensure enterprise healthcare compliance (HIPAA/GDPR) by sanitizing Personal Identifiable Information (PII/PHI) and implementing guardrails against prompt injection and jailbreaks.

## 2. Core Strategies
- **Pre-pipeline PII Masking**: Using regex and Presidio to redact names, emails, phones, and SSNs before payloads reach external LLM endpoints.
- **Guardrails Frameworks**: NeMo Guardrails or Guardrails AI to enforce topical boundaries.
- **Private VPC Deployments**: Hosting open-source models (Llama 3, Mistral) on AWS SageMaker or GCP Vertex AI for regulated domains.

## 3. Code Walkthrough (`src/interview_prep/evaluator.py`)
```python
sanitized = sanitize_pii("Patient John Doe (john.doe@hospital.org)")
# Returns "Patient John Doe ([EMAIL_REDACTED])"
```

## 4. Interview Narrative
"In healthcare and biotech consulting, data privacy is paramount. I enforce a Zero Trust architecture with pre-pipeline PII masking using Microsoft Presidio and strict guardrails before any prompt hits our foundation models."
