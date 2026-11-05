# VECANOVA AI Interview Prep — Plataforma de preparación y aceleración de carrera (From Zero to Hero)

[English](./README.md) | [Español](./README.es.md)

> **Domina las entrevistas técnicas, el system design y el LLMOps para roles de Senior AI/ML Product Engineer en 5 industrias clave.**

---

## 🌟 Visión general

Este repositorio provee un framework de preparación riguroso y de grado producción para ingenieros de Machine Learning e IA Senior que apuntan a posiciones remotas de alta compensación en consultoras tecnológicas globales y ingeniería enterprise. Cubre **5 industrias clave**:
1. **Healthcare & Biotech** (cumplimiento HIPAA/GDPR, sanitización de PII/PHI, RAG clínico)
2. **Fintech & Banking** (detección de fraude, detección de anomalías, cumplimiento PCI-DSS)
3. **E-commerce & Retail** (sistemas de recomendación, filtrado colaborativo, embeddings vectoriales)
4. **Logistics & Supply Chain** (forecasting de series temporales, mantenimiento predictivo, optimización de rutas)
5. **SaaS & Enterprise Tech** (búsqueda vectorial multi-tenant, LLMOps, sistemas multi-agente con LangGraph)

---

## 🚀 Google Colab y CLI interactivo ("From Zero to Hero")

Puedes ejecutar la masterclass completa de forma interactiva en Google Colab o vía nuestro CLI interactivo:

- **Notebook de Google Colab**: sube `notebooks/ai_interview_masterclass.ipynb` a [Google Colab](https://colab.research.google.com).
- **CLI de entrevistas simuladas interactivas**:
  ```bash
  make install
  make mock       # Ejecuta una sesión de entrevista simulada interactiva
  ```

---

## 🏗️ Arquitectura de System Design y banco de preguntas STAR

- **Diagrama de arquitectura**: [Pipeline RAG y MLOps empresarial multi-industria](./docs/es/architecture.md)
- **Banco de respuestas STAR**: [Respuestas STAR curadas en español](./docs/es/star_bank.md) · [versión en inglés](./docs/star_bank.md)

---

## 📚 Fundamentos teóricos (`docs/es/theory/`)

1. [Teoría 1: Fundamentos avanzados de ML (métricas, árboles, detección de anomalías, embeddings)](./docs/es/theory/01-advanced-machine-learning-foundations.md)
2. [Teoría 2: IA generativa y arquitectura RAG avanzada](./docs/es/theory/02-generative-ai-and-rag-architecture.md)
3. [Teoría 3: LLMOps, evaluaciones (RAGAS) y observabilidad](./docs/es/theory/03-llmops-evaluations-and-observability.md)
4. [Teoría 4: Seguridad de IA en salud, PII y guardrails](./docs/es/theory/04-healthcare-ai-security-and-guardrails.md)
5. [Teoría 5: System Design y dominio de STAR](./docs/es/theory/05-system-design-and-behavioral-star.md)
6. [Teoría 6: Arquitecturas de IA multi-industria (Healthcare, Fintech, E-commerce, Logistics, SaaS)](./docs/es/theory/06-multi-industry-ai-architectures.md)

## 🗺️ Tutoriales paso a paso (`docs/tutorials/`)

Los tutoriales ejecutables están en inglés (los comandos y salidas que verás son en inglés):

1. [Tutorial 1: De ML tradicional a GenAI](./docs/tutorials/01-from-zero-to-hero-ml-genai.md)
2. [Tutorial 2: LLMOps, evaluaciones y observabilidad (RAGAS)](./docs/tutorials/02-llmops-evaluations-ragas.md)
3. [Tutorial 3: Guardrails de IA, PII y cumplimiento en salud](./docs/tutorials/03-healthcare-guardrails-pii.md)
4. [Tutorial 4: System Design y dominio de STAR (en inglés)](./docs/tutorials/04-system-design-star-mastery.md)

---

## 📦 Arquitectura y estándares
Siguiendo los estándares de **Anvil Engineering** y la metodología **Software Factory**:
- Desarrollo guiado por especificaciones (`docs/specs/`)
- Architecture Decision Records (`docs/adr/`)
- Contratos funcionales (`docs/contracts/`)
- Memoria por capas (`.memory/`)
