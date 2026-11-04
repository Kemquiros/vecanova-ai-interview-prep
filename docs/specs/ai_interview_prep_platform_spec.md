# Specification: AI Interview Prep & Evaluation Platform

## 1. Objective
Build an engineering platform and curriculum to prepare Senior ML/AI Engineers for technical interviews and production AI roles in healthcare/biotech.

## 2. Requirements & Acceptance Criteria
- **Requirement 1**: Provide PII sanitization for clinical text.
  - *Given* patient clinical notes containing names and IDs, *When* processed through the sanitizer, *Then* sensitive PII/PHI is masked before external LLM calls.
- **Requirement 2**: Provide RAG evaluation metrics simulation (Faithfulness, Answer Relevance).
  - *Given* a generated response and retrieved context, *When* evaluated, *Then* return numerical scores and audit logs.
- **Requirement 3**: Interactive CLI assessment tool.
  - *Given* candidate prompts, *When* executing mock assessments, *Then* provide structured feedback and STAR interview prompts.
