# Arquitectura de IA empresarial y pipeline RAG

> Referencia visual para entrevistas de System Design: plataforma RAG y MLOps de salud end-to-end.
> Se complementa con `docs/theory/05-system-design-and-behavioral-star.md` (presupuestos de latencia, deep dives)
> y `docs/theory/06-multi-industry-ai-architectures.md` (tablas de restricciones por industria).

```mermaid
graph TD
    A[Client Application / UI] -->|HTTPS / JWT Auth| B[API Gateway & Rate Limiter]
    B --> C[PII / PHI Sanitizer (Presidio / Regex)]
    C -->|Sanitized Payload| D[LangGraph Orchestrator (Multi-Agent DAG)]

    D -->|1. Tabular Feature Request| E[Traditional ML Engine (XGBoost / Isolation Forest)]
    D -->|2. Semantic & Keyword Query| F[Hybrid Retriever (Dense Embeddings + BM25)]

    F --> G[(Vector DB & Document Store)]
    G -->|Top-k Chunks| H[Cohere Cross-Encoder Reranker]

    H -->|Reranked Context + Tabular Features| I[Foundation Model (Private VPC / AWS SageMaker)]

    I -->|Generated Response| J[Guardrails & Output Validation (NeMo)]
    J -->|Audited Response| K[RAGAS Evaluation & Langfuse Observability]
    K --> A

    F -.->|Fallback: BM25-only| F
    H -.->|Skip on timeout: RRF order| I
    I -.->|Secondary provider / cached answers| J
    C -.->|FAIL CLOSED on sanitizer failure| B
```

Las aristas sólidas son el camino feliz; las punteadas son los caminos de degradación de la tabla de modos de fallo de abajo.

## 1. Justificación componente por componente

Por qué existe cada caja y qué se rompe sin ella — esta es la ronda de "justifica tu diagrama" del system design:

| Componente | Por qué está ahí | Qué se rompe sin él |
|---|---|---|
| API Gateway & Rate Limiter | Terminación de AuthN/Z, token buckets por tenant, moldeado de requests | Cualquier tenant puede starving el pool; no hay un único lugar para imponer cuotas o soltar carga por tier |
| Sanitizador PII/PHI | Puerta bloqueante fail-closed antes de cualquier llamada a modelo | PHI llega a proveedores/trazas/logs — un incidente de cumplimiento, no un bug |
| Orquestador LangGraph | DAG multi-paso determinista con checkpointing y reintentos | Cadenas ad-hoc: sin recuperación de estado, sin replay, branching no testeable, fallos parciales silenciosos |
| Motor de ML tradicional | Scoring tabular/fraude en latencia de ms, determinista y explicable (SHAP) | Todo forzado por el LLM: 50x latencia, 100x costo, decisiones no reproducibles |
| Retriever híbrido (denso + BM25) | El denso atrapa paráfrasis, BM25 atrapa identificadores exactos (códigos de fármacos, números de error) | Solo léxico: pierde paráfrasis; solo denso: pierde IDs exactos y términos raros |
| Vector DB & Document Store | ANN a escala (HNSW), aislamiento por namespace de tenant, filtros de metadata | Fuerza bruta en memoria: escaneos O(n), sin aislamiento, sin escala horizontal |
| Reranker Cross-Encoder | Precisión en el corte del top-k: +8-10 puntos de precisión vs RRF crudo | Selección de contexto más débil -> ruido en el prompt -> más alucinación y costo de tokens |
| Foundation Model (VPC) | Generación con control contractual de datos | Exposición de PHI/PCI a subprocessors del proveedor; la auditoría no puede avalar el camino de datos |
| Guardrails de salida (NeMo) | Re-escaneo de PII, enforcement de políticas y disclaimers, checks de respuesta a inyección | Una inyección exitosa o un fallo del sanitizador llega textual al usuario |
| Eval RAGAS + Langfuse | Traza cada nodo; tendencias de faithfulness/precision; atribución de costo por tenant | Regresiones de calidad encontradas por usuarios, no por dashboards; el costo por tenant queda sin respuesta |
| Caché semántica/de respuesta (edge, scoped por tenant) | Recorte de 25-40% del gasto LLM en tráfico repetido, mejora de TTFT | El costo escala 1:1 con el crecimiento; cachés sin clave de versión filtran respuestas obsoletas o cross-tenant |

## 2. Planificación de capacidad (números trabajados)

Supuesto de tráfico: 100k queries/día; la hora pico concentra el 12% del volumen diario.

```
avg QPS            = 100k / 86,400           ~= 1.2 QPS
peak sustained QPS = 100k x 0.12 / 3,600     ~= 3.3 QPS
design point       = 15 RPS  (4-5x de headroom de ráfaga para reintentos, picos, jobs batch)
concurrent LLM generations = 15 RPS x 2.4s   = 36 streams
KV-cache RAM (clase 7B, ctx 4k, ~0.5 GB/stream)  = 36 x 0.5 = 18 GB
                   -> 2x A100-80GB o 4x A10G-24GB réplicas (pesos + KV + headroom)
```

**Dimensionamiento del Vector DB** (la matemática que te piden hacer en el whiteboard):

```
vectores crudos      = 10M chunks x 1536 dims x 4 B (fp32) = 61.4 GB
overhead HNSW        x 1.5-2.0 (M=16-32 enlaces de grafo/capa) = ~92-123 GB
plan de nodos        = 4-6 nodos de 32 GB, o 3 nodos en fp16 (30.7 GB crudos -> ~46-61 GB)
con HA (RF=2)        -> 6-8 nodos clase r6g.xlarge (~$600-1,800/mes, ver tabla de costos)
compresión           = PQ a 64 B/vector -> 0.64 GB (96x) con un costo de recall de 0.85-0.93
presupuesto de ingesta = 20k docs/día x 900 tokens; re-embed del corpus completo en ~9h a ~310 vec/s
```

**Throughput de tokens**: 100k x (2,400 prompt + 350 completion) = 240M tokens de prompt + 35M de completion por día. Los tokens de prompt dominan el 87% del gasto — por eso el diseño de prefijos amigable con prompt-cache y el caching semántico le ganan a la optimización de elección de modelo como primera palanca de costo.

## 3. Modos de fallo y degradación elegante

| Fallo | Detección | Camino de degradación | Impacto al usuario | Nunca hacer |
|---|---|---|---|---|
| Vector DB caído | Health check / tasa de error | Retrieval solo-BM25; embeddings cacheados para chunks calientes | Recall cae ~15-25%, sigue respondiendo | Enviar queries crudas al LLM sin retrieval |
| Reranker caído | Timeout > 200ms | Saltar rerank, servir orden RRF | Precisión -8-10 pts | Bloquear el request por una etapa no crítica |
| Proveedor LLM caído | Tasa de 5xx / circuit breaker | Proveedor secundario -> respuestas cacheadas -> encolar y notificar | Más lento o pregrabado para algunas queries | Hard-fail sin comunicación al tenant |
| Sanitizador caído | Health check | **FAIL CLOSED**: rechazar el request | Sin respuesta (cumplimiento primero) | Puentear la puerta "temporalmente" |
| Pico de latencia de embeddings | Monitor p99 en la etapa de embed | Reducir top-k / servir índice fp16 en disco | Leve reducción de recall | Encolar sin límite — backpressure o shedding |
| Observabilidad caída | Errores de buffer | Ring buffer fire-and-forget, replay al recuperarse | Ninguno | Hacer el tracing una dependencia síncrona |
| Caché semántica obsoleta | Mismatch de clave de versión de contenido | Purgar por versión, puentear caché | Respuestas correctas, mayor costo | Invalidación solo por TTL en contenido regulado |

Regla de diseño que codifica la tabla: **la estrategia de fallo sigue a la clase de cumplimiento** — el sanitizador falla cerrado (healthcare), mientras que retriever, reranker y proveedor fallan abiertos con fallbacks (disponibilidad).

## 4. Presupuesto de latencia (p95, copiloto clínico con streaming, SLA de 2.5s)

Cada etapa recibe una asignación explícita; la suma debe cumplir el SLA con headroom. Re-ejecuta esta tabla cada vez que un feature agregue una etapa.

| Etapa | Presupuesto p95 | Notas |
|---|---|---|
| AuthN/Z + gateway + rate limit | 20 ms | Verificación JWT local, sin salto de red extra |
| Sanitizador PII (Presidio) | 80 ms | Analyzers cacheados, nota de 2k tokens |
| Retrieval híbrido (HNSW + BM25, RRF) | 300 ms | Réplicas calientes, misma AZ |
| Reranker (top-25) | 120 ms | Scoring batched del cross-encoder |
| LLM TTFT | 700 ms | Prefill de 2.4k tokens, streaming activo |
| Generación LLM (350 tokens) | 1,100 ms | ~320 tok/s efectivos |
| Guardrails de salida | 100 ms | Solo checks determinísticos inline |
| **Total** | **~2.42s** | SLA de 2.5s, ~3% de headroom |

Reglas transversales: los checks pesados (juez RAGAS, escaneos profundos de red team) nunca van en el camino crítico; los p95 por etapa se emiten como spans para que una violación de presupuesto sea atribuible a una etapa, no adivinada; y el presupuesto es un artefacto de producto — los features negocian contra él en planeación, no en producción.

## 5. Modelo de datos y aislamiento por tenant

| Store | Mecanismo de aislamiento | Forma de clave |
|---|---|---|
| Vector DB | Namespace/partición por tenant | `tenant:{id}:docs:{version}` |
| Document store (S3) | Prefijo + bucket policy por tenant | `s3://docs/{tenant_id}/{doc_id}/` |
| Postgres (estado del orquestador, caché) | Row-level security sobre tenant_id | toda tabla lleva `tenant_id` |
| Caché de respuesta/semántica | Tenant + versión de política + versión de contenido en la clave | `tenant:{id}:pv:{v}:cv:{v}:qh:{hash}` |
| Telemetría (Langfuse) | tenant_id en cada span; redactar antes de exportar | trace -> etiqueta de tenant obligatoria |

El aislamiento se testea como la autenticación: el CI incluye intentos de acceso cross-tenant contra cada store, y un test de aislamiento fallido es un bloqueador de release — no un ítem del backlog.

Objetivos de nivel de servicio que esta arquitectura compromete (y la puerta que hace cumplir cada uno):

- Faithfulness >= 0.95 en el gold set estratificado — puerta de release en CI (RAGAS batch + proxy inline con `evaluate_rag_response`).
- Recall del sanitizador PII >= 0.99 contra el set adversarial congelado — puerta de CI + PSI semanal sobre distribuciones de entidades.
- End-to-end p95 <= 2.5s, p99.9 monitoreado para regresiones de cola (merges, cachés frías) — alertas sobre spans.
- Aislamiento por tenant: cero lecturas cross-tenant — tests adversariales en CI contra cada store, bloqueador de release en caso de fallo.

## 6. Modelo de costos (ejemplo: 100k queries/día = 3M/mes)

Supuestos declarados para que el entrevistador pueda desafiarlos: 2,400 tokens de prompt + 350 de completion por llamada; $3/$15 por 1M tokens; 40% de los tokens de prompt caen en un prompt cache del proveedor con 50% de descuento; 30% de hit rate de caché semántica sobre el resto sin cachear.

| Componente | Base de dimensionamiento | Costo mensual |
|---|---|---|
| Inferencia LLM (baseline sin caché) | 7.2B in x $3/1M + 1.05B out x $15/1M | $37,350 |
| — con prompt cache (40% a mitad de precio) | in: $21,600 x 0.8 = $17,280 + out: $15,750 | $33,030 |
| — con caché semántica (30% de llamadas servidas completas) | x 0.70 | **~$26,100** |
| Reranker (hospedado, $2.00 por 1k búsquedas) | 3M búsquedas/mes | $6,000 (o ~$400 self-hosted A10G) |
| Vector DB | 4 x r6g.xlarge x $0.201/hr x 730h | ~$590 (RF=2: ~$1,180) |
| Tier de aplicación (gateway, orquestador, guards) | 3 x c7g.2xlarge x ~$0.294/hr x 730h | ~$645 |
| Embeddings (ingesta + queries, self-hosted @ $0.10/1M) | ~720M tokens/mes | ~$72 |
| Observabilidad (Langfuse self-hosted + Postgres) | 2 nodos pequeños | ~$400 |
| Storage (docs en S3, backups) | ~1 TB + versionado | ~$80 |
| **Total (ajustado)** | | **~$34k/mes, de los cuales ~75% es inferencia LLM** |

Lee la tabla como una priorización: la eficiencia de tokens (diseño de prompts, caching, recorte de contexto) mueve el número más que todas las líneas de infraestructura combinadas. El `src/interview_prep/cost.py` de este repo (`estimate_tokens`, `estimate_cost`, `monthly_projection`) es la referencia de aritmética; `docs/theory/05` muestra la variante de 50k/día ($19.8k/mes) derivada con él.

## 7. Etapas de escalado

| Etapa | Tráfico | Qué cambia |
|---|---|---|
| MVP (0->1) | < 1 RPS | API única + proveedor administrado + nodo único de índice vectorial; observabilidad desde el día uno (barata), aislamiento por namespace |
| Producto (1->10 RPS) | 100k/día | Reranker, caché semántica, cuotas por tenant, réplicas HNSW calientes, cadena de fallback de proveedor |
| Plataforma (10->100 RPS) | 1M+/día | Vector DB sharded (RF=2), carril de ingesta dedicado, modelos self-hosted o híbridos por tier de costo, pipeline de eval batch, libro de costos por tenant |

Referencias cruzadas: `src/interview_prep/evaluator.py` (`sanitize_pii`, `evaluate_rag_response`), `src/interview_prep/drift.py` (`population_stability_index`, `ks_statistic`, `drift_verdict`), `src/interview_prep/cost.py` (`estimate_tokens`, `estimate_cost`, `monthly_projection`).
