# Teoría 5: System Design y dominio behavioral con STAR

> Versión en español del documento original en inglés: [05-system-design-and-behavioral-star.md](../../theory/05-system-design-and-behavioral-star.md)

## Respuestas relámpago para la entrevista

1. "Yo diseño hacia atrás desde el presupuesto de latencia: cada etapa recibe una asignación
   p95, y las asignaciones deben sumar por debajo del SLA —si no suman, corto alcance, no el
   SLA."
2. "La calidad del retrieval es el techo de la calidad de la respuesta —yo tuneo recall@k antes
   de tocar una sola palabra de un prompt."
3. "Cada componente tiene una ruta de degradación: reranker caído significa saltarse el rerank,
   vector DB caído significa solo BM25, sanitizer caído significa fail closed."
4. "Una historia STAR sin números es una opinión —yo siempre llevo métricas before/after con
   unidades."
5. "El caching compra latencia y costo, pero en dominios regulados la key de caché debe llevar
   tenant, versión de política y versión de contenido —o se convierte en un incidente de
   compliance."

## 1. La forma de la entrevista de System Design de 60 minutos

Para prompts como "Diseña un copiloto de análisis clínico para 50k queries/día":

1. **Requisitos (5 min)**: funcionales (RAG sobre protocolos clínicos, enmascaramiento de PII,
   citas) y no funcionales (SLA de latencia p95, disponibilidad, clase de compliance, techo de
   costo mensual). Di la clase de compliance en voz alta —dicta la estrategia de fallos (fail
   closed para HIPAA, fail open con fallback para e-commerce).
2. **Estimaciones (5 min)**: deriva QPS y volumen de tokens. 50k queries/día = 0.58 QPS
   promedio, pero los picos diurnos concentran 10-15% del volumen diario en una hora -> ~2 QPS
   sostenidos en pico, diseña para 10-15 RPS con headroom de burst/retry. Tokens: 50k x
   (2,400 prompt + 400 completion) = 120M de tokens de prompt + 20M de completion por día -> a
   $3/$15 por 1M de tokens eso es $660/día, ~$19,800/mes. Cítalo —la literacidad de costos
   separa al senior del mid.
3. **API + modelo de datos (10 min)**: `POST /analyze` (streaming SSE), contrato de ingesta de
   documentos, modelo de tenants.
4. **Diseño de alto nivel (15 min)**: refleja el diagrama de referencia en `docs/architecture.md`.
5. **Deep dives (15 min)**: elige 1-2 en los que el entrevistador se meta (escalado del índice
   vectorial, caching, multi-tenancy, manejo de fallos). Ten listas las tablas de trade-off de
   este documento.
6. **Transversales (5 min)**: observabilidad, seguridad, atribución de costos, matriz de
   degradación.

## 2. La tabla de presupuesto de latencia (memoriza una de estas)

Copiloto clínico, SLA extremo a extremo p95 de 2.5s (streamed):

| Etapa | Presupuesto p95 | Notas |
|---|---|---|
| AuthN/Z + gateway + rate limit | 20 ms | Verificación JWT local, sin salto de red |
| Sanitizer de PII (NER de Presidio) | 80 ms | Analyzers cacheados; nota de 2k tokens |
| Recuperación híbrida (HNSW top-50 + BM25 top-50, RRF) | 300 ms | Índice caliente, misma AZ |
| Reranker cross-encoder (top-25) | 120 ms | Scoring en batch |
| LLM time-to-first-token | 700 ms | Prefill de 2.4k tokens, streaming activado |
| Generación completa del LLM (350 tokens) | 1,100 ms | ~320 tok/s efectivos |
| Guardrails de salida (re-scan de PII + política) | 100 ms | Solo chequeos deterministas inline |
| **Total** | **~2.42s** | Bajo 2.5s con ~3% de headroom |

Frase de entrevista: "Cuando alguien pide agregar una feature, vuelvo a correr esta tabla. Las
features son gratis hasta que rompen la suma."

## 3. Deep Dive 1 (tabla de trade-off): escalado del índice vectorial

Primero las matemáticas de sizing —sale en cada entrevista:

```
raw_bytes   = n_chunks x dim x bytes_per_scalar
            = 10M x 1536 x 4 B = 61.4 GB (fp32)
HNSW RAM    ~= raw x 1.5-2.0 (graph links, M=16-32 neighbors/layer)
            = ~92-123 GB  ->  4-6 nodes of 32 GB, or 3 nodes at fp16
fp16 raw    = 30.7 GB; PQ 64 B/vector = 0.64 GB (96x compression, recall cost)
shards      = ceil(hnsw_ram / usable_ram_per_node)
```

Luego la tabla de trade-off entre familias de índice:

| Índice | Build (10M x 1536d) | Query p95 | Recall@10 | RAM vs crudo | Mejor para |
|---|---|---|---|---|---|
| HNSW | 30-60 min | 5-15 ms | 0.95+ | 1.5-2.0x | Latencia crítica, escala moderada |
| IVF-PQ | 10-20 min | 3-8 ms | 0.85-0.93 | ~0.1x | Alto QPS, sensible a costo |
| DiskANN | horas | 10-30 ms (NVMe) | ~0.95 | ~0.1x + disco | Escala de mil millones, RAM escasa |
| ScaNN | minutos-horas | rápido | alto | medio | Amigable para research, con apetito de tuning |

Knobs de tuning que hay que nombrar: `M` (grado del grafo: 16 default, 32-48 para recall alto),
`ef_construction` (calidad del build), `ef_search` (dial de recall/latencia en query-time —
expónlo como config, no como constante). Menciona los segment merges: los inserts en streaming
sobre HNSW provocan merges en background que hacen picos en el p99 —agenda las ingestas masivas
y dimensiona réplicas solo para tráfico de búsqueda.

## 4. Deep Dive 2 (tabla de trade-off): capas de caching

| Capa | Hit rate típico | Riesgo de corrección | Diseño de key | Política de TTL |
|---|---|---|---|---|
| Caché de respuesta exacta | 5-15% | Solo staleness | tenant + policy_ver + content_ver + query_hash | Purge en actualización del índice |
| Caché semántica | 20-40% | Riesgo de respuesta errada (casi-duplicado != misma pregunta) | Scoped por tenant, threshold coseno >= 0.97 | Corta (horas), por dominio |
| Provider prompt (KV) cache | n/a | Ninguno | Prefijo estable: system prompt + few-shot primero | Recorta TTFT 30-60% en prefijos compartidos |
| Caché de embeddings | ~100% en repeticiones | Ninguno | Hash de contenido del chunk | Infinita (content-addressed) |

Regla de dominios regulados: las cachés semánticas sobre contenido clínico/financiero necesitan
una eval —yo corrí un gold set de 500 queries a través de una caché con threshold de 0.97 y medí
una tasa de respuestas erradas de 1.4%; eso mató la idea para dosificaciones pero la mantuvo
para contenido de nivel FAQ. Contar esa historia (con el número) es exactamente la profundidad
que las entrevistas senior buscan.

## 5. Deep Dive 3 (tabla de trade-off): evaluación streaming vs batch

| Dimensión | Eval inline (streaming) | Eval batch (offline) |
|---|---|---|
| Bloquea una respuesta mala | Sí — chequeos deterministas baratos | No — detecta después del hecho |
| Profundidad de métricas | Proxies de faithfulness, re-scan de PII, chequeos de rechazo | Faithfulness RAGAS completa con juez LLM, revisión humana |
| Costo | Llamadas al juez en el camino crítico | 10-50x más barato, en batch |
| Semántica de fallo | Fail closed o degradar con banner | Pager, dataset en cuarentena, gate de regresión para el próximo release |

El repo implementa el proxy inline en `src/interview_prep/evaluator.py`:

```python
from interview_prep.evaluator import evaluate_rag_response

verdict = evaluate_rag_response(query, response, context_chunks)
if verdict["status"] == "REVIEW_REQUIRED":     # faithfulness or relevance < 0.5
    ...  # degrade: add disclaimer, offer human handoff, log to eval dataset
```

Las corridas batch de RAGAS van semanales sobre una muestra estratificada más cada ítem
REVIEW_REQUIRED; el CI bloquea releases cuando la media de faithfulness del gold set cae más de
2 puntos.

## 6. Dominio behavioral: STAR con números

- **Situation (20-30s)**: contexto, escala, stakes. "Copiloto clínico en una red de 2,000
  clínicos, 50k queries/día, quejas de alucinaciones llegando al CMO."
- **Task (1 oración)**: la meta medible. "Bajar la tasa de alucinaciones por debajo de 2% sin
  romper el presupuesto p95 de retrieval de 1s."
- **Action (60-70% de la historia)**: declaraciones en "yo", decisiones *con las alternativas
  rechazadas y por qué*. Aquí vive la seniority —los juniors listan acciones, los seniors
  justifican trade-offs.
- **Result**: 2-3 métricas cuantificadas + un cambio durable. "Alucinaciones 12% -> 1.8%, p99
  850ms -> 620ms, y un gate de faithfulness en CI que atrapó 3 regresiones en el trimestre
  siguiente."

El patrón de la **escalera de métricas** conecta métricas de sistema con resultados de negocio
—a los entrevistadores se les quedan las escaleras, no los números sueltos:

```
+12% clinician weekly adoption   (business)
  <- CSAT 3.9 -> 4.5             (product)
    <- hallucination 12% -> 1.8% (quality)
      <- p99 retrieval 850ms -> 620ms  (system)
```

Anti-patrones que se leen a junior: "nosotros" sin "yo" (propiedad poco clara), resultados sin
unidades ni baselines, culpar a los predecesores, y alcance sin límites ("mejoré todo").

El banco completo de preguntas con 15 respuestas modelo y follow-ups de entrevistador vive en
`docs/star_bank.md`.

## 7. Referencias cruzadas al código del repo

Detección de drift sobre el comportamiento de servicio (`src/interview_prep/drift.py`), por
ejemplo monitoreando distribuciones de latencia o de longitud de query semana a semana:

```python
from interview_prep.drift import (
    population_stability_index, ks_statistic, drift_verdict,
)

psi = population_stability_index(last_week_p95s, this_week_p95s, bins=10)
if drift_verdict(psi) == "SEVERE":   # > 0.25
    ...  # page the on-call; freeze deploys until explained
ks = ks_statistic(expected, actual)  # nonparametric complement
```

Proyección de costos para capacity planning (`src/interview_prep/cost.py` — firma según el
módulo):

```python
from interview_prep.cost import estimate_tokens, estimate_cost, monthly_projection

n_in  = estimate_tokens(question + " " + " ".join(chunks))   # chars/4 heuristic
per_call = estimate_cost(n_in, tokens_out=350)
print(monthly_projection(per_call, calls_per_day=50_000))    # ~$19.8k/mo at $3/$15
```

## 8. Historias de guerra en producción / Errores frecuentes

1. **La caché semántica sirvió respuestas de protocolo desactualizadas durante 3 días después
   de una actualización de guía** —la key no llevaba versión de contenido. Mitigación: keys con
   versión estampada + purge automático en la actualización del índice; las cachés para
   contenido regulado se invalidan por eventos, nunca solo por TTL.
2. **El p99 se veía bien en dashboards, pero 1 de cada 50 requests tardaba 4s** —merges de
   segmentos de HNSW tras ingestas masivas. Mitigación: ventanas dedicadas de ingesta, réplicas
   solo de búsqueda, topología de un solo escritor; alerta sobre p99.9, no sobre p99.
3. **El checkpoint store de LangGraph (Postgres) se volvió el cuello de botella a 15 QPS** —
   escrituras de estado síncronas por nodo. Mitigación: checkpointing async con staleness
   acotado, tablas particionadas, y retención de checkpoints de 24h (los estábamos guardando
   para siempre sin razón de recuperación).
4. **Evals offline en verde, faithfulness de producción cayó 14 puntos** —la distribución de
   queries se movió después de que una clínica nueva se incorporó (templates de notas
   distintos). Mitigación: PSI sobre embeddings y longitudes de queries como gate semanal; sets
   de eval refrescados trimestralmente desde estratos de tráfico en vivo, no congelados para
   siempre.
5. **Un cambio de prompt agregó un bloque few-shot y creció el costo 38% en silencio** —
   detectado en una revisión mensual, no por alertas. Mitigación: presupuestos de tokens por
   feature con lint en CI sobre los templates de prompt (techo de caracteres/token), y alertas
   de `monthly_projection` sobre deltas de costo semana a semana > 10%.

## 9. Lo que los entrevistadores realmente preguntan

**Q1: "Diseña un copiloto clínico para 50k queries/día con restricciones HIPAA."**
Respuesta modelo: recorre la forma de 6 pasos de la sección 1; aterriza la estrategia de fallos
guiada por compliance, la tabla de presupuesto de latencia y las matemáticas de ~$20k/mes de
tokens sin que te lo pidan.

**Q2: "¿Cómo pondrías el p99 del retrieval por debajo de 1 segundo?"**
Respuesta modelo: "Réplicas HNSW calientes para tráfico solo de búsqueda, ef_search tuneado
contra un objetivo de recall, colocación en la misma AZ, índice fp16 o PQ si estás limitado por
RAM, top-k dimensionado para el reranker (50 -> 25), y caché de embeddings para chunks
repetidos. Después reviso la cola: el p99.9 normalmente señala merges o cachés frías, no la
ruta de la query."

**Q3: "¿Es segura una caché semántica en un entorno regulado?"**
Respuesta modelo: "Solo si es por tenant, con versión estampada, con gate por threshold y
evaluada. Yo cuantifico la tasa de respuestas erradas sobre un gold set antes de activarla; para
contenido cercano a dosificaciones la mantengo exact-match. La caché es una optimización, nunca
una capa de corrección."

**Q4: "Cuéntame un fallo de producción que hayas sido dueño."**
Respuesta modelo: elige de la categoría 1 de `docs/star_bank.md`; abre con la detección,
cuantifica el blast radius, y cierra con el fix sistémico (lo que no puede regresar).

**Q5: "¿Cómo evalúas un sistema RAG antes del lanzamiento?"**
Respuesta modelo: "Un gold set de 200-500 ítems construido desde logs reales de queries,
faithfulness RAGAS + context precision/recall, calibración humana sobre 50 ítems para hacer
sanity check al juez LLM (kappa de Cohen juez-humano > 0.7 o el juez no se confía), más proxies
inline vía `evaluate_rag_response` en staging. Gate de lanzamiento: faithfulness >= 0.95 en el
gold set, y cada ítem REVIEW_REQUIRED triageado."

**Q6: "¿Cómo aíslas tenants en una plataforma RAG compartida?"**
Respuesta modelo: "Data plane: namespaces por tenant (Pinecone namespaces / particiones de
Milvus), tenant en cada key de caché, row-level security en Postgres. Control plane: token
buckets y cuotas por tenant. Economía: cada trace etiquetada con tenant_id para que el costo por
tenant sea una query, no un proyecto. Los bugs de aislamiento son Sev-1 —yo testo acceso
cross-tenant en CI igual que testeamos auth."

**Q7: "¿Cuándo haces fine-tuning en vez de RAG + prompting?"**
Respuesta modelo: "Orden por default: prompt -> tools/orquestación -> RAG -> fine-tune. El
fine-tune justifica su costo cuando necesito formato/comportamiento consistentes a escala,
latencia de un modelo más pequeño destilado, o estilo de dominio —no para conocimiento (ese es
el trabajo del índice, que se refresca). La vara: ya medí que prompt+RAG falla sobre un set
etiquetado, y puedo nombrar la métrica que el fine-tune mueve y su plan de datos de
entrenamiento."

## 10. Lecturas recomendadas

- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- RAGAS documentation: https://docs.ragas.io/
- Langfuse documentation: https://langfuse.com/docs
- Paper de HNSW — Malkov & Yashunin: https://arxiv.org/abs/1603.09320
- Paper de DiskANN: https://arxiv.org/abs/2105.09613
- Google SRE Book (SLOs, error budgets): https://sre.google/sre-book/table-of-contents/
- Designing Data-Intensive Applications (O'Reilly): https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/
- Cohere Rerank guide: https://docs.cohere.com/docs/rerank-guide

Referencias cruzadas: `src/interview_prep/evaluator.py` (`evaluate_rag_response`), `src/interview_prep/drift.py` (`population_stability_index`, `ks_statistic`, `drift_verdict`), `src/interview_prep/cost.py` (`estimate_tokens`, `estimate_cost`, `monthly_projection`).
