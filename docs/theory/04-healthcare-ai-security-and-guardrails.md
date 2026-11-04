# Theory 4: Healthcare AI Security, PII, and Guardrails

## 1. Regulatory Compliance in Healthcare (`HIPAA`, `GDPR`)
- Protection of Protected Health Information (PHI) and Personally Identifiable Information (PII) is mandatory.
- Zero-tolerance policy for data leaks to public commercial LLM APIs without strict Business Associate Agreements (BAAs) or enterprise privacy guarantees.

## 2. PII / PHI Sanitization Pipelines
- **Named Entity Recognition (NER)** & Regular Expressions: Identifying names, Social Security Numbers, medical record numbers, phone numbers, and emails.
- **Masking & Anonymization**: Replacing detected entities with tokens (`[PATIENT_REDACTED]`, `[EMAIL_REDACTED]`) prior to model ingestion and unmasking post-generation only for authorized users.

## 3. AI Guardrails (`NeMo Guardrails`, `Guardrails AI`)
- **Input Guardrails**: Blocking prompt injections, jailbreaks, toxic language, and out-of-scope domain queries before they reach the LLM.
- **Output Guardrails**: Verifying that generated responses adhere to safety policies, do not fabricate medical diagnoses without disclaimers, and do not leak internal system instructions.
