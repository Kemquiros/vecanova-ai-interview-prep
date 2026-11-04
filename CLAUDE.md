# CLAUDE.md - VECANOVA AI Interview Prep Charter

> Project-specific configuration and engineering standards for `vecanova-ai-interview-prep`.
> Extends the Master Constitution (`/home/kemquiros/Documentos/Comando/CLAUDE.md`).

---

## 🎯 Purpose and Scope

This repository provides a rigorous, production-grade preparation framework for Senior ML/AI Engineers targeting high-compensation roles in global healthcare, fintech, and enterprise AI consulting (e.g., Senior AI/ML Product Engineer positions). It bridges traditional predictive machine learning with modern Generative AI, LLMOps, advanced RAG architectures, AI guardrails, PII/PHI protection, and System Design (STAR methodology in English).

---

## 📐 Engineering Standards

- **Language Policy**: Technical documentation, code, tests, specs, and memory are written in **English**. Live interaction with users is in Spanish.
- **TDD (Test-Driven Development)**: All evaluation modules, sanitizers, and retrieval pipelines start with failing unit/integration tests (`pytest`).
- **SDD (Spec-Driven Development)**: Specifications in `docs/specs/`, ADRs in `docs/adr/`, contracts in `docs/contracts/`.
- **Quality Gates**: 100% test coverage on core evaluation logic, strict typing (`pyright`/`mypy`), and clean lint (`ruff`).

---

## 📁 Repository Layout

```
vecanova-ai-interview-prep/
├── .memory/                    # Layered memory architecture
│   ├── episodic/               # Session history & milestones
│   ├── semantic/               # Interview frameworks, STAR templates, concepts
│   └── procedural/             # Mock interview runbooks & practice scripts
├── docs/                       # Specifications, ADRs, contracts
│   ├── specs/                  # SDD feature specs
│   ├── adr/                    # Architecture Decision Records
│   └── contracts/              # Functional interfaces
├── src/                        # Core interview prep & evaluation code
│   └── interview_prep/         # Evaluators, guardrails, retrieval logic
├── tests/                      # Testing pyramid (unit, integration)
├── scripts/                    # CLI tools & mock assessment runners
├── KANBAN.md                   # Project Kanban board
├── README.md                   # Public landing and navigation
└── pyproject.toml              # Dependencies & tooling config
```
