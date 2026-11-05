# Teoría 2: Generative AI y arquitectura RAG avanzada

> Versión en español del documento original en inglés: [02-generative-ai-and-rag-architecture.md](../../theory/02-generative-ai-and-rag-architecture.md)

La mitad de arquitectura del loop senior: retrieval-augmented generation como pipeline
multi-etapa en lugar de una demo de vector store, recuperación híbrida con fusión con
principios, reranking, la decisión RAG-versus-fine-tuning, y orquestación multi-agente bajo
restricciones de compliance. La posición senior: el retriever, no el LLM, es donde los sistemas
RAG fallan en producción, así que destina la mayor parte del esfuerzo de tuning a chunking,
retrieval y reranking.

## Respuestas relámpago para la entrevista

- "La recuperación híbrida no es un nice-to-have: BM25 atrapa identificadores exactos —códigos
  de error, números de protocolo, SKUs— que los modelos densos difuminan dentro de la prosa
  circundante, mientras que la recuperación densa atrapa paráfrasis que BM25 no ve. Reciprocal
  rank fusion los combina sin ninguna calibración de scores."
- "El chunking es una decisión de calidad de retrieval, no un paso de procesamiento de texto:
  elijo el tamaño de chunk y el overlap midiendo recall@k sobre un golden set, no por
  convención."
- "RAG para conocimiento, fine-tuning para comportamiento: no puedes llegar con prompts ni con
  retrieval a un formato de salida consistente o a un estilo de dominio privado, y no deberías
  hacer fine-tuning para memorizar documentos que cambian cada semana."
- "Modelo los workflows de agentes como state graphs explícitos con checkpoints y gates
  humanos, no como loops libres —el determinismo es un requisito de compliance en salud y
  fintech, no una preferencia."
- "El reranking con cross-encoder es la etapa de mayor ROI del pipeline: puntúa query y
  documento en conjunto, corta 50 candidatos al top 3, y típicamente compra ganancias de doble
  dígito en calidad de respuesta por unos 100 milisegundos."

## 1. RAG como pipeline multi-etapa

Un pipeline de producción es: ingesta, chunking, recuperación híbrida, fusión, reranking,
generación aterrizada y evaluación. Cada etapa tiene un contrato y un modo de fallo.

**Chunking.** Las ventanas de palabras de tamaño fijo con overlap preservan los límites
contextuales a través de los cortes. Con `chunk_size=64` palabras y `overlap=16` (los defaults
en `src/interview_prep/retrieval.py`), el stride es de 48 palabras, así que un documento de $W$
palabras rinde aproximadamente:

$$\text{chunks} \approx 1 + \left\lceil \frac{W - \text{chunk\_size}}{\text{chunk\_size} - \text{overlap}} \right\rceil$$

El tamaño de chunk intercambia granularidad de retrieval por coherencia de contexto: los chunks
pequeños rankean con precisión pero dejan al generador sin contexto; los grandes mantienen el
significado intacto pero diluyen la señal tanto para el embedder como para el LLM. Las ventanas
fijas trituran tablas y código; el split recursivo sobre separadores estructurales (encabezados,
párrafos, oraciones) o el parsing consciente de estructura mantiene las unidades atómicas
enteras, y el patrón parent-document recupera chunks pequeños pero le entrega al generador su
sección padre.

**Recuperación híbrida.** BM25 puntúa documentos por la frecuencia de los términos de la query
saturada por $k_1$ y normalizada por la longitud del documento vía $b$:

$$\text{score}(q, d) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot
\frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \left(1 - b + b \cdot \frac{\lvert d \rvert}{\text{avgdl}}\right)}$$

$$\text{IDF}(q_i) = \ln\left(\frac{N - df(q_i) + 0.5}{df(q_i) + 0.5} + 1\right)$$

donde $N$ es el tamaño del corpus, $df(q_i)$ la frecuencia documental del término $q_i$, y
$f(q_i, d)$ su frecuencia en $d$. Los defaults $k_1 = 1.5$, $b = 0.75$ son estándar y rara vez
vale la pena tunearlos antes de tener un eval set de retrieval. La recuperación densa embebe
query y pasajes con un bi-encoder y busca en un índice ANN (HNSW da latencia casi logarítmica
con overhead de memoria de grafo; IVF-PQ intercambia recall por memoria vía quantization).

| Dimensión | BM25 (sparse) | Dense embeddings | Híbrido + RRF |
| --- | --- | --- | --- |
| Identificadores exactos (códigos, SKUs, números de protocolo) | Excelente | Débil | Excelente |
| Paráfrasis y match semántico | Débil | Excelente | Excelente |
| Vocabulario nuevo sin reentrenar | Inmediato | Requiere re-embedding | Inmediato |
| Calibración de scores entre queries | Escala TF-IDF, comparable entre queries | Coseno, dependiente del modelo | Fusión solo por rangos, sin calibración necesaria |
| Infraestructura | Índice invertido, barato | Índice ANN, hambriento de memoria | Ambos, más un paso de fusión |
| Fallo característico | Vocabulary mismatch | Misses confiados sobre entidades raras | Un ranker dominando la fusión |
| Cuándo elegirlo | Corpora legales/clínicos con mucho exact-match | Dominios exploratorios, ricos en sinónimos | Default para RAG empresarial |

**Fusión.** Reciprocal rank fusion combina rankings usando solo los rangos, que es por lo que
no necesita calibración de escalas de score entre retrievers heterogéneos:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60$$

La constante $k = 60$ amortigua el dominio de las posiciones rank-1, así que un ítem que
aparece a media lista en ambos rankings le gana a un ítem rankeado primero en solo uno —un
efecto de voto por mayoría que es exactamente lo que quieres al fusionar BM25 con un ranker
denso.

**Reranking.** Un cross-encoder atiende en conjunto el par query-documento y produce un score
de relevancia. No puede pre-indexar, así que corre solo sobre la shortlist.

| Dimensión | Bi-encoder (recuperación densa) | Cross-encoder (reranker) |
| --- | --- | --- |
| Scoring | Embeddings independientes, coseno | Atención conjunta sobre query + documento |
| Latencia | Milisegundos a escala ANN | Decenas de milisegundos por par |
| Escala de candidatos | Millones | Cientos, tras el prefilter |
| Techo de recall | Más bajo | Más alto |
| Rol en el pipeline | Generación de candidatos | Ordenamiento final antes de generar |

## 2. Patrones RAG avanzados que vale la pena nombrar

- **Transformación de queries:** reescritura, expansión o HyDE (genera una respuesta
  hipotética, embebe eso) para cerrar la brecha de vocabulario query-documento.
- **Filtrado por metadata:** rangos de fechas, tipo de documento y —crítico— aislamiento por
  tenant, forzado en la capa de query, no con instrucciones en el prompt.
- **Compresión contextual:** extracción post-retrieval del span relevante de cada chunk para
  gastar la ventana de contexto en señal.
- **Encabezados contextuales de chunks:** antepone el título del documento y la ruta de sección
  a cada chunk antes de embeber; barato y consistentemente subestimado.
- **Caching y canaries:** cachea mapeos query-a-contexto para queries frecuentes; corre un set
  fijo de preguntas canary en cada cambio de índice o de modelo.

## 3. RAG vs fine-tuning vs long-context

| Dimensión | RAG | Fine-tuning (PEFT/LoRA) | Prompting long-context |
| --- | --- | --- | --- |
| Frescura del conocimiento | Re-indexa en cualquier momento | Congelado al entrenar; refrescar implica reentrenar | Sube el documento por request |
| Citas y auditabilidad | Procedencia por chunk | Ninguna | A nivel documento como mucho |
| Comportamiento, estilo, formato | Débil | Fuerte | Débil a medio |
| Costo marginal a escala | Infra de retrieval + tokens por llamada | Entrenamiento único; prompts más livianos | La cuenta de tokens crece con el contexto |
| Privacidad de datos | El corpus queda en tu VPC; embebe localmente | Los datos entran al pipeline de entrenamiento | El documento completo sale de tu frontera |
| Latencia | Agrega un salto de retrieval | No agrega | Prefill largo |
| Elígelo cuando | Documentos cambiantes, procedencia requerida | Estilo estable, terminología, adherencia a schema | Análisis único de un documento grande |

El orden senior es: prompt engineering primero, RAG cuando los hechos son externos y
cambiantes, fine-tuning al final —cuando el cuello de botella es el comportamiento, no el
conocimiento.

**LoRA** hace barata la opción de fine-tuning: congela los pesos preentrenados
$W_0 \in \mathbb{R}^{d \times k}$ y aprende una actualización de bajo rango:

$$h = W_0 x + \Delta W x = W_0 x + B A x, \qquad
A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r},\ r \ll \min(d, k)$$

Los parámetros entrenables bajan de $d \cdot k$ a $r(d + k)$ —con $r = 8$ sobre un modelo 7B
eso es aproximadamente el 0.1 por ciento de los pesos. La actualización se escala por $\alpha/r$,
se aplica típicamente a las proyecciones de attention, y QLoRA agrega una base congelada
quantizada a 4 bits para entrenar en una sola GPU. El fine-tuning cambia comportamiento; no
agrega conocimiento recuperable de forma confiable, y no puede citar nada.

## 4. Sistemas multi-agente y gestión de estado (LangGraph)

- **Máquinas de estado explícitas por encima de loops libres:** el workflow es un grafo de
  estado tipado, funciones de nodo y aristas condicionales. Los ciclos están permitidos pero
  acotados por un recursion limit, lo que convierte "el agente se fue a pasear" de un misterio
  en un error de configuración.
- **Checkpointing:** el estado durable por super-paso habilita time-travel debugging, reanudar
  después de un fallo y replay para auditorías —un requisito duro en industrias reguladas.
- **Gates con humano en el loop:** interrumpe antes de tools con efectos secundarios (escribir
  en un EHR, ejecutar mutaciones SQL, mover dinero), serializa el estado pendiente y reanuda
  con la aprobación. Diseña estos como nodos de grafo de primera clase, no como manejo de
  excepciones ad-hoc.
- **Topología multi-agente:** un supervisor/router que delega en subgrafos especialistas escala
  mejor que la charla peer-to-peer entre agentes; cada salto entre agentes es un punto
  potencial de amplificación de errores, así que usa por defecto la menor cantidad de agentes
  que resuelvan el problema.
- **Cuándo no construir agentes:** si una sola llamada al LLM más un tool loop alcanza,
  despliega eso. Cada frontera de agente agrega latencia, costo y una clase nueva de fallo.

## Referencia cruzada con el repositorio

El pipeline anterior está implementado en `src/interview_prep/retrieval.py` — tokenización,
BM25 sobre chunks, fusión RRF y chunking con overlap:

```python
from interview_prep.retrieval import (
    BM25Index,
    chunk_text,
    reciprocal_rank_fusion,
    tokenize,
)

chunks = chunk_text(clinical_protocol, chunk_size=64, overlap=16)  # stride 48
index = BM25Index(chunks)

query = "sepsis bundle lactate remeasurement"
sparse_ids = [doc_id for doc_id, _ in index.search(query, top_k=20)]
dense_ids = [doc_id for doc_id, _ in dense_search(embed(query), top_k=20)]

fused = reciprocal_rank_fusion([sparse_ids, dense_ids], k=60)[:5]
```

La tokenización es deliberadamente minimal —minúsculas, sin puntuación, split por espacios en
blanco:

```python
tokenize("Anomaly Detection, TELEMENTRY!")  # ["anomaly", "detection", "telemetry"]
```

La calidad de la generación se hace gate post-hoc en `src/interview_prep/evaluator.py`:

```python
from interview_prep.evaluator import evaluate_rag_response, sanitize_pii

safe_query = sanitize_pii(
    "Patient jdoe@hospital.org, SSN 123-45-6789, asks about metformin dosing"
)
report = evaluate_rag_response(query=safe_query, response=draft_answer, context=fused_chunks)
if report["status"] != "PASSED":  # requires faithfulness >= 0.5 and answer_relevance >= 0.5
    escalate_to_review(report["reason"])
```

`sanitize_pii` enmascara emails, teléfonos y SSNs antes de que nada llegue a un modelo de
embeddings o a un endpoint de LLM; `evaluate_rag_response` computa faithfulness por proxy
léxico y relevancia de respuesta por respuesta (la versión production-grade de estas métricas
se cubre en la Teoría 3).

## Historias de guerra en producción / Errores frecuentes

- **Tablas trituradas por chunking de tamaño fijo.** Una tabla de referencia de laboratorio
  quedó partida a media fila; el retrieval devolvió media tabla y el modelo inventó la otra
  mitad. Causa raíz: tamaño de chunk ciego a la estructura del documento. Mitigación: parsing
  consciente de estructura que mantenga las tablas atómicas, chunk headers con título y ruta de
  sección, y el patrón parent-document para el contexto de generación.
- **Fusión híbrida dominada por un ranker.** El blending ponderado por score dejó que los
  scores coseno densos (rango 0.6-0.9) ahogaran los scores a escala IDF de BM25, así que los
  códigos de error exactos nunca salieron a la superficie. Causa raíz: fusionar escalas de
  score incomparables. Mitigación: RRF basado en rangos (como en `reciprocal_rank_fusion`) en
  lugar de blending de scores.
- **La latencia del reranker reventó el SLO.** Un cross-encoder sobre 100 candidatos a ~120 ms
  por par agregó 12 segundos. Causa raíz: sin análisis del presupuesto de candidatos.
  Mitigación: rerankea solo el top 20-50, limita la shortlist hacia atrás desde el SLO de
  latencia, cachea scores para queries frecuentes.
- **Índice desactualizado sirviendo guías viejas.** Una guía clínica fue revisada pero el
  índice de embeddings siguió sirviendo el chunk viejo; las evals offline "pasaban" porque el
  golden set codificaba el texto viejo. Causa raíz: sin versionado del índice. Mitigación:
  versiona el índice junto con el corpus, re-indexa en eventos de cambio de documentos y corre
  queries canary por release.
- **PHI embebido crudo en un vector store compartido.** Los resultados de kNN filtraron el
  contexto de pacientes vecinos entre tenants. Causa raíz: sanitización omitida antes de
  embeber, y el índice no tenía frontera por tenant. Mitigación: `sanitize_pii` en la ingesta,
  namespaces por tenant más metadata filters forzados, y control de acceso deny-by-default.
- **Modelo de embeddings cambiado sin re-indexar.** El lado de queries fue actualizado, el lado
  del corpus no; la similitud coseno entre espacios mismatched devolvía basura con apariencia
  plausible. Causa raíz: los embeddings de query y de corpus tratados como desplegables
  independientes. Mitigación: fija la versión del modelo de embeddings al índice y despliega
  ambos como una unidad.

## Lo que los entrevistadores realmente preguntan

- **"Diseña un sistema RAG para una base de conocimiento clínica."**
  - Aislamiento multi-tenant primero (namespaces, filtros forzados, audit logs), luego
    recuperación híbrida con RRF, rerank con cross-encoder, generación con citas forzadas.
  - Menciona sin que te lo pidan la evaluación (context recall, faithfulness) y el versionado
    del índice.
- **"¿Por qué recuperación híbrida en vez de puramente densa?"**
  - Identificadores exactos, actualizaciones de vocabulario sin reentrenar y fusión sin
    calibración.
  - Da la fórmula de RRF y explica el amortiguado de $k = 60$.
- **"¿Cómo eliges el tamaño de chunk?"**
  - Es una decisión empírica: construye un set de preguntas doradas, mide recall@k y nDCG@k a
    través de tamaños, respeta las fronteras estructurales y usa overlap para preservar el
    contexto de los bordes.
  - Chunks pequeños para QA de respuesta corta; retrieval más grande y jerárquico para
    síntesis de formato largo.
- **"¿Cuándo harías fine-tuning en vez de RAG?"**
  - Comportamiento, estilo, adherencia a schema, latencia/costo a escala —con conocimiento
    estable.
  - Las matemáticas de LoRA como evidencia de dominio técnico: $r(d+k)$ parámetros entrenables
    en lugar de $d \cdot k$.
- **"¿Cómo evitas que un agente haga daño?"**
  - Grafo de estado acotado con recursion limits, checkpoints, allow-lists de tools, gates
    HITL sobre efectos secundarios, tools idempotentes y logging de traces completo.
- **"Las respuestas del RAG son fluidas pero erróneas. Depúralo."**
  - Descompón primero: ¿se recupera el contexto correcto (context recall) o la generación
    ignora buen contexto (faithfulness)?
  - Arregla el retriever antes de tocar prompts; agrega comportamiento de rechazo para queries
    de baja precisión de contexto.
- **"¿Cómo aíslas tenants en un vector store compartido?"**
  - Opciones en fuerza creciente: metadata filters en la capa de query, namespaces por tenant,
    índices por tenant; control de acceso forzado fuera del LLM, más cifrado y audit trails.

## Lecturas recomendadas

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
  (NeurIPS 2020) — https://arxiv.org/abs/2005.11401
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
  (Foundations and Trends in Information Retrieval, 2009) — https://dl.acm.org/doi/10.1561/1500000019
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP
  2020) — https://arxiv.org/abs/2004.04906
- Nogueira and Cho, "Passage Re-ranking with BERT" (2019) — https://arxiv.org/abs/1904.08398
- Malkov and Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using
  Hierarchical Navigable Small World Graphs" (IEEE TPAMI, 2018) — https://arxiv.org/abs/1603.09320
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022) —
  https://arxiv.org/abs/2106.09685
- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023)
  — https://arxiv.org/abs/2312.10997
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023) —
  https://arxiv.org/abs/2210.03629
- LangGraph documentation (state graphs, checkpointing, human-in-the-loop) —
  https://langchain-ai.github.io/langgraph/
