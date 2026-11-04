import sys
from interview_prep.evaluator import sanitize_pii, evaluate_rag_response

def main():
    print("=== VECANOVA AI Interview Prep: Interactive Mock Assessment ===")
    sample_query = "What is the patient ID for john.doe@biotech.com and how do we apply RAGAS?"
    print(f"\n[Raw Query]: {sample_query}")
    
    sanitized = sanitize_pii(sample_query)
    print(f"[Sanitized Query (PII Protected)]: {sanitized}")
    
    context = [
        "RAGAS metrics include Faithfulness, Answer Relevance, and Context Precision.",
        "Patient data must be anonymized before LLM processing."
    ]
    response = "RAGAS metrics include Faithfulness and Answer Relevance for auditing model outputs."
    
    eval_result = evaluate_rag_response(sanitized, response, context)
    print(f"\n[RAG Evaluation Result]: {eval_result}")
    print("\nAssessment simulation completed successfully.")

if __name__ == "__main__":
    main()
