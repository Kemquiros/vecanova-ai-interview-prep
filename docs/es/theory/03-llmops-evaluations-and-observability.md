# Teoría 3: LLMOps, evaluaciones y observabilidad

> Versión en español del documento original en inglés: [03-llmops-evaluations-and-observability.md](../../theory/03-llmops-evaluations-and-observability.md)

La mitad de operaciones del loop senior: cómo los sistemas generativos se evalúan con
honestidad, se trazan, se controlan en costo y se monitorean por drift después del lanzamiento.
La posición senior: las métricas NLP estáticas son canaries, no veredictos; un único score de
calidad mezclado oculta qué etapa del pipeline se rompió; y los dos números que te despiertan
con el pager a las 3 a.m. son la latencia p99 y el costo por cada mil requests.

## Respuestas relámpago para la entrevista

- "BLEU y ROUGE miden overlap de strings; los usuarios no. Yo descompongo la calidad —
  faithfulness para el aterrizaje, answer relevance para la utilidad, context precision y
  context recall para el retriever— porque un único score mezclado oculta qué etapa se rompió."
- "RAGAS faithfulness es la fracción de claims atómicos de la respuesta que están entailados
  por el contexto recuperado: claims soportados sobre claims totales. Es una tasa de
  alucinación con definición legal."
- "Yo despliego con latencia p50/p95/p99 y costo de tokens por request desde el día uno: la
  media esconde la cola, y la cola es lo que dispara el pager."
- "LLM-as-judge es un modelo, no una métrica: lo valido contra etiquetas humanas con
  estadísticos de acuerdo antes de confiar en él, aleatorizo la posición de las respuestas para
  matar el position bias, y fijo la versión del juez como cualquier otro artefacto de modelo."
- "Los thresholds de PSI son mi contrato permanente de drift: bajo 0.1 estable, de 0.1 a 0.25
  en observación, sobre 0.25 severo —el mismo contrato tanto si la señal es un score de fraude
  como un embedding de RAG."

## 1. Por qué las métricas NLP estáticas fallan en sistemas generativos

BLEU es una media geométrica de precisiones de n-gramas recortadas contra referencias;
ROUGE-L es una F-measure sobre la subsecuencia común más larga:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right) \qquad
\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot P_{lcs} \cdot R_{lcs}}{\beta^2 P_{lcs} + R_{lcs}}$$

Ambas comparan strings contra una referencia, así que premian redacciones vecinas a la
paráfrasis, colapsan en generación abierta con muchas respuestas válidas y son juegables con la
longitud. Su rol honesto restante es el de canaries de regresión sobre formatos de salida
fijos (extractores, clasificadores con salidas con template) —nunca calidad frente al usuario.

| Familia de métricas | Qué mide | Costo | Correlación con humanos | Uso apropiado |
| --- | --- | --- | --- | --- |
| BLEU / ROUGE | Overlap de n-gramas vs referencia | Despreciable | Pobre en generación abierta | Canary de regresión en formatos fijos |
| BERTScore | Similitud semántica de tokens vía embeddings | Moderado | Mejor | QA de respuesta corta |
| Familia RAGAS | Faithfulness, relevancia, métricas de contexto | Moderado | Buena para pipelines RAG | Gate de CI para RAG |
| LLM-as-judge | Adherencia a rúbrica, preferencia por pares | Alto | Alta con rúbrica y debiasing | Gates de calidad con matices, ranking por pares |

## 2. La descomposición RAGAS

RAGAS (Retrieval Augmented Generation Assessment) divide la calidad de RAG en métricas locales
por etapa. **Faithfulness** descompone la respuesta en claims atómicos $S$ y verifica cada uno
contra el contexto recuperado, produciendo el conjunto de claims soportados $V$:

$$F = \frac{\lvert V \rvert}{\lvert S \rvert}$$

**Answer relevance** genera preguntas candidatas $\hat{q}_i$ a partir de la respuesta y puntúa
qué tanto cada una embebe de vuelta hacia la pregunta original $q$:

$$\text{AR} = \frac{1}{N} \sum_{i=1}^{N} \cos(q, \hat{q}_i)$$

**Context precision** premia los chunks relevantes rankeados temprano —con $v_i \in \{0, 1\}$ el
indicador de relevancia del chunk en el rank $i$ y precision@$i$ la proporción de chunks
relevantes en el top $i$:

$$\text{CP} = \frac{1}{k} \sum_{i=1}^{k} \text{precision}@i \cdot v_i$$

**Context recall** pregunta qué proporción de las afirmaciones ground-truth son atribuibles al
contexto recuperado. El mapeo por etapas es el poder diagnóstico:

| Métrica RAGAS | Etapa del pipeline que acusa | Qué significa un score malo |
| --- | --- | --- |
| Context recall | Retriever o cobertura del corpus | Documentos faltantes, mal troceados o desactualizados |
| Context precision | Fusión y ranking | El ruido le gana a la señal en la shortlist |
| Faithfulness | Generación y prompt de grounding | El modelo inventa más allá del contexto |
| Answer relevance | Entendimiento de la query y generación | Respuestas evasivas, verbosas o fuera de objetivo |

Aterriza las métricas antes de confiar en ellas: un proxy léxico (overlap de palabras entre
respuesta y contexto, como el implementado en `src/interview_prep/evaluator.py`) es barato,
determinista y amigable con CI; un juez LLM es semántico pero caro y sensible a la versión.
Despliega el proxy en CI, muestrea el juez offline, y reconcilia ambos sobre un set de
auditoría etiquetado.

## 3. LLM-as-Judge, tratado como un modelo

- **Pointwise vs pairwise:** la puntuación pointwise con rúbrica da números absolutos y
  dashboardables; la preferencia pairwise es más confiable para "cuál salida es mejor" pero
  produce scores relativos. Usa pairwise para selección de modelos, pointwise para
  monitoreo.
- **Sesgos conocidos:** position bias (gana la respuesta mostrada primero —mitiga alternando
  el orden y promediando), verbosity bias (ganan las respuestas más largas —controla por
  longitud), y self-preference (un juez favorece salidas de su propia familia de modelos —
  elige un juez de una familia distinta a la del generador).
- **Validación:** antes de confiar en un juez, mide el acuerdo con etiquetas humanas. El kappa
  de Cohen corrige el acuerdo crudo $p_o$ por el acuerdo al azar $p_e$:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- **Gobernanza:** fija la versión del modelo juez, re-baselinea los scores en cualquier
  upgrade del juez, y loguea las rationales del juez para auditabilidad.

## 4. Observabilidad y tracing

- **Traces estructuradas, no logs:** una trace por request, un span por etapa del pipeline
  (retrieve, rerank, generate, tool call), con atributos compatibles con OpenTelemetry para que
  sean portables entre vendors. El trace ID debe ligar el registro de eval con el prompt exacto,
  los chunks recuperados y la versión del modelo —ese vínculo es lo que hace posible el
  debugging.
- **Contabilidad de tokens:** prompt vs completion vs tokens cacheados por etapa. El costo de
  tokens es una dimensión de SLO de primera clase; una feature que triplica el contexto triplica
  la cuenta en silencio salvo que los presupuestos sean por etapa.
- **Percentiles de latencia:** reporta p50/p95/p99. Para un pipeline de $k$ etapas con colas
  independientes, el p99 extremo a extremo queda aproximadamente acotado por la suma de los p99
  por etapa (union bound):

$$p99_{e2e} \lesssim \sum_{j=1}^{k} p99_j$$

  que es por qué los presupuestos de latencia por etapa deben asignarse de arriba hacia abajo
  desde el SLO que ve el usuario, no medirse de abajo hacia arriba después del hecho.
- **Taxonomía de errores:** distingue timeouts del proveedor, fallos de tools, rechazos de
  guardrails y respuestas de baja calidad con status 200. Solo la última aparece en los
  dashboards de latencia; las cuatro pertenecen al dashboard de calidad.
- **Sampling:** captura del 100 por ciento de traces para errores y requests lentos,
  head- o tail-based sampling en el resto; guarda para siempre los atributos relevantes para
  eval, y los payloads solo mientras el compliance lo permita.

## 5. Ingeniería de costos

La estimación de tokens usa la heurística de cuatro-caracteres-por-token con un piso de uno,
que es suficientemente precisa para presupuestar texto en inglés:

$$\hat{t} = \max\left(1, \left\lceil \frac{\lvert \text{text} \rvert}{4} \right\rceil\right)$$

El costo mezclado separa el pricing de input y de output (los tokens de output se cobran más
caro):

$$\text{cost} = \frac{t_{in}}{10^6} \cdot p_{in} + \frac{t_{out}}{10^6} \cdot p_{out}$$

Proyección mensual sobre un mes de 30 días:

$$\text{monthly} = 30 \cdot r \cdot c$$

con $r$ requests por día y $c$ costo por request. Ejemplo resuelto: 1M de tokens de input a
\$3/MTok más 500K de tokens de output a \$15/MTok es \$3.00 + \$7.50 = \$10.50; 100 requests
por día a \$0.01 cada uno proyecta \$30.00 al mes. Las palancas, en orden de impacto: caching
exacto y semántico de retrievals y completions frecuentes, model routing (modelo pequeño por
default con escalado por complejidad), recorte de contexto y reuso del system prompt, y batch
APIs para todo lo offline.

## 6. Monitoreo de drift para sistemas LLM

El Population Stability Index sobre $B$ bins, con proporciones de referencia $e_i$ y
proporciones en vivo $a_i$:

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i) \ln \frac{a_i}{e_i}$$

con el contrato implementado en `src/interview_prep/drift.py`: PSI por debajo de 0.1 es STABLE,
de 0.1 a 0.25 es MODERATE, por encima de 0.25 es SEVERE. El estadístico KS lo complementa:

$$D_{n,m} = \sup_x \lvert F_{1,n}(x) - F_{2,m}(x) \rvert$$

| Dimensión | PSI | Estadístico KS |
| --- | --- | --- |
| Salida | Suma ponderada de log-ratios sobre bins | Máximo hueco de la CDF |
| Sensibilidad | Dependiente de la resolución de bins; bins gruesos ocultan shifts concentrados | Detecta shifts locales en cualquier punto del soporte |
| Regla de decisión | Thresholds de regla empírica (0.1 / 0.25) | Valores críticos de la distribución nula dependiente del tamaño de muestra |
| Ubicación típica | Dashboards sobre features y scores | Test formal sobre la distribución del score o del embedding |
| En este repo | `population_stability_index` + `drift_verdict` | `ks_statistic` |

Para RAG específicamente, monitorea tres capas: la distribución de queries (embebe las queries
entrantes, sigue la distancia de centroides y las participaciones por cluster), la calidad del
retrieval (context recall sobre un set rotatorio de preguntas canary), y la calidad de la salida
(faithfulness muestreado y juzgado offline). Un cambio de versión del modelo por parte de un
proveedor también es drift: fija versiones, y trata cualquier alias sin fijar como una
dependencia sin monitorear.

## Referencia cruzada con el repositorio

Gate de evaluación post-generación en `src/interview_prep/evaluator.py`:

```python
from interview_prep.evaluator import evaluate_rag_response, sanitize_pii

safe_query = sanitize_pii(raw_query)  # redact PHI/PII before any logging or tracing
report = evaluate_rag_response(query=safe_query, response=answer, context=top_chunks)
# {"faithfulness": 0.83, "answer_relevance": 0.9, "status": "PASSED", ...}
gate_open = report["status"] == "PASSED"  # both metrics >= 0.5
```

Detección de drift en `src/interview_prep/drift.py`:

```python
from interview_prep.drift import drift_verdict, ks_statistic, population_stability_index

psi = population_stability_index(baseline_scores, live_scores)
verdict = drift_verdict(psi)  # "STABLE" | "MODERATE" | "SEVERE"
if verdict == "SEVERE":
    d = ks_statistic(baseline_scores, live_scores)  # localize where the CDF diverges
```

Estimación y proyección de costos en `src/interview_prep/cost.py`:

```python
from interview_prep.cost import estimate_cost, estimate_tokens, monthly_projection

tokens_in = estimate_tokens(prompt_text)      # ~4 chars per token, floor of 1
usd = estimate_cost(
    tokens_in=tokens_in, tokens_out=400,
    price_in_per_mtok=3.0, price_out_per_mtok=15.0,
)
annual_usd = 12 * monthly_projection(requests_per_day=5000, cost_per_request=usd)
```

## Historias de guerra en producción / Errores frecuentes

- **Prompt bloat como regresión de costo silenciosa.** Una feature "ayudante" triplicó el
  contexto inyectado; la factura subió \$40k/mes sin ningún cambio funcional. Causa raíz: sin
  presupuesto de tokens por etapa y sin alertas de costo. Mitigación: `estimate_tokens` en CI
  sobre prompts representativos, alerta sobre p95 de tokens por request, y recorte de contexto
  por relevancia.
- **Deprecación silenciosa del modelo del proveedor.** Un alias fijado a latest fue
  retirado; las completions cambiaron de formato y de pricing de la noche a la mañana. Causa
  raíz: tratar el modelo del proveedor como una dependencia estable. Mitigación: fija versiones
  exactas, contract tests sobre prompts dorados en CI, tráfico canary en cualquier bump de
  versión, fallback dual-provider.
- **Overfitting al golden set.** El equipo iteró prompts hasta que el score de eval llegó a
  0.98 en 50 casos mientras las quejas reales de usuarios subían. Causa raíz: el set de eval se
  filtró en las iteraciones de prompts y era demasiado pequeño para ser representativo.
  Mitigación: un holdout congelado que el loop de iteración nunca ve, al menos unos cientos de
  casos, refresco trimestral desde tráfico en vivo, y etiquetado a ciegas.
- **Juez que cambia de opinión entre versiones.** Los dashboards mostraron un swing de calidad
  de 12 puntos que seguía a un upgrade del modelo juez, no al producto. Causa raíz: el juez se
  trató como ground truth y nunca se versionó. Mitigación: fija el juez, re-baselinea en cada
  upgrade, y sigue el acuerdo juez-vs-humano de forma continua.
- **Traces que filtraron PHI a un SaaS de terceros.** Queries crudas de pacientes aterrizaron
  en una herramienta de observabilidad alojada. Causa raíz: sin redacción en la frontera del
  span. Mitigación: `sanitize_pii` antes de emitir cualquier atributo, redacción dentro de la
  propia capa de instrumentación, self-hosting o vendors cubiertos por DPA para workloads
  regulados, y muestreo de auditoría de lo que realmente se almacena.
- **Un SLO de p50 enmascarando un desastre de p99.** La latencia mediana era de 800 ms mientras
  el p99 llegaba a 9 segundos por un fan-out del reranker. Causa raíz: el SLO se definió sobre
  la mediana. Mitigación: presupuestos de p99 por etapa asignados con el union bound, alertas
  de latencia de cola, y load tests que reportan percentiles, no medias.

## Lo que los entrevistadores realmente preguntan

- **"¿Cómo evalúas una feature LLM de extremo a extremo?"**
  - Por capas: unit tests deterministas, un set de eval offline (métricas RAGAS más juez con
    rúbrica), canaries de regresión en CI, y un A/B online con métricas guardrail.
  - Agrega muestreo de auditoría humana y di qué haces cuando las capas discrepan.
- **"¿Qué es exactamente faithfulness y cómo lo calcularías?"**
  - Descomposición en claims más entailment: $F = \lvert V \rvert / \lvert S \rvert$.
  - Contrasta el proxy léxico (barato, determinista, CI) con el juez LLM (semántico, caro,
    sensible a la versión) y nombra cuándo usas cada uno.
- **"¿Cómo mitigas prompt injection a través del contenido recuperado?"**
  - Trata el texto recuperado como datos, nunca como instrucciones: delimita, instruye al
    modelo en consecuencia, mantiene permisos de tools con mínimo privilegio, filtra salidas, y
    ponle gate HITL a las acciones con efectos secundarios.
- **"Diseña la observabilidad de un pipeline LLM multi-etapa."**
  - Spans por etapa con un trace ID compartido, atributos de token/costo/latencia por etapa,
    una taxonomía de errores que incluya los 200 de baja calidad, redacción de PII en la
    frontera, y alertas sobre p99 y costo por cada 1k requests.
- **"¿Cómo sabes que tu set de eval sirve de algo?"**
  - Representatividad frente al tráfico en vivo, dispersión de dificultad, acuerdo entre
    anotadores, estabilidad en el tiempo, disciplina de holdout, y una cadencia de refresco.
- **"Los costos subieron 5x después del lanzamiento. Guíame por el diagnóstico."**
  - Contabilidad de tokens por etapa (`estimate_tokens`), mezcla de requests por segmento, tasa
    de aciertos de caché, mezcla de modelos, creep de contexto, loops de retry/timeout; luego
    palancas rankeadas por esfuerzo e impacto (caching, routing, recorte, batch).
- **"¿Cuándo importa más el drift que la exactitud?"**
  - Sistemas de decisión soportada y regulados donde la degradación es silenciosa; la respuesta
    es la arquitectura de monitoreo: PSI/KS sobre inputs y outputs, monitoreo de calibración,
    sets de preguntas canary, y una política de rollback que de verdad se haya ensayado.

## Lecturas recomendadas

- Es, James, Jin, Coca, and Nakov, "RAGAS: Automated Evaluation of Retrieval Augmented
  Generation" (EACL 2024) — https://arxiv.org/abs/2309.15217
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023) —
  https://arxiv.org/abs/2306.05685
- Papineni, Roukos, Ward, and Zhu, "BLEU: A Method for Automatic Evaluation of Machine
  Translation" (ACL 2002) — https://aclanthology.org/P02-1040/
- Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (Text Summarization
  Branches Out, 2004) — https://aclanthology.org/W04-1013/
- Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (ICLR 2020) —
  https://arxiv.org/abs/1904.09675
- Post, "A Call for Clarity in Reporting BLEU Scores" (WMT 2018) —
  https://aclanthology.org/W18-6319/
- Langfuse documentation (LLM observability and evals) — https://langfuse.com/docs
- LangSmith documentation (tracing, datasets, evaluation) — https://docs.smith.langchain.com/
- OpenTelemetry Traces concept docs — https://opentelemetry.io/docs/concepts/signals/traces/
