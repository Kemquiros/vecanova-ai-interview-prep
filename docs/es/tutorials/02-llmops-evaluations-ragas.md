# Tutorial 2: LLMOps — de BM25 a retrieval híbrido a evaluación estilo RAGAS

> Versión en español del documento original en inglés: [02-llmops-evaluations-ragas.md](../02-llmops-evaluations-ragas.md)

> Para quién es: ingenieros preparando entrevistas de AI/ML Senior que necesitan
> demostrar un loop completo de calidad de retrieval-augmented generation (RAG): retrieval
> léxico, chunking, ranking híbrido con reciprocal rank fusion, y evaluación automatizada
> estilo RAGAS cableada como puerta de regresión. Presupuesto de tiempo: 75 minutos.

## Lo que vas a construir

1. Un índice de búsqueda BM25 sobre un pequeño corpus de conocimiento clínico.
2. Un chunker que convierte un documento de política largo en chunks RAG con solapamiento.
3. Un retriever híbrido que fusiona un ranking denso simulado con BM25 vía reciprocal
   rank fusion (RRF).
4. Un loop de evaluación estilo RAGAS (`evaluate_rag_response`) que puntúa faithfulness y
   answer relevance, más una puerta de regresión de golden set que podrías enchufar en CI.
5. Observabilidad de tokens y costos para el pipeline.

## El pipeline en una línea

```text
documents -> chunk_text -> BM25Index + (simulated) dense ranker
          -> reciprocal_rank_fusion -> top-k context -> response
          -> evaluate_rag_response (faithfulness, answer_relevance) -> CI gate
```

## Paso 1: Prepara el entorno

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -c "from interview_prep.retrieval import BM25Index; from interview_prep.evaluator import evaluate_rag_response; print('imports ok')"
```

Salida esperada:

```text
imports ok
```

**Variante de Google Colab:** abre `notebooks/ai_interview_masterclass.ipynb` en Colab y
ejecuta en una celda nueva:

```python
%cd /content/vecanova-ai-interview-prep
!pip install -e ".[dev]"
```

Vuelve a ejecutar la instalación una vez por sesión de Colab; el entorno es efímero.

## Paso 2: Tokeniza e indexa el corpus con BM25

Crea un corpus pequeño de cinco documentos de una línea mezclando conocimiento clínico y
de retrieval (mantenlo diminuto para poder predecir los rankings a mano):

```python
from interview_prep.retrieval import tokenize, BM25Index

docs = [
    "Metformin is the first-line medication for type 2 diabetes. The typical starting dose is 500 mg twice daily with meals.",
    "Hypertension is diagnosed when systolic blood pressure stays above 140 mmHg. Lifestyle changes are tried before medication.",
    "BM25 is a bag-of-words ranking function that scores documents using term frequency and inverse document frequency.",
    "Vector search embeds documents and queries into dense vectors and retrieves nearest neighbors by cosine similarity.",
    "Reciprocal rank fusion combines multiple ranked lists by summing 1 divided by k plus rank for every document.",
]

print(tokenize("Dose of Metformin for Type 2 Diabetes"))

index = BM25Index(docs)
print(index.search("metformin dose diabetes", top_k=3))
```

Salida esperada:

```text
['dose', 'of', 'metformin', 'for', 'type', '2', 'diabetes']
[(0, 3.8687)]
```

Dos cosas que notar. Primero, `tokenize` pasa a minúsculas y quita puntuación, así que
"Metformin" y "metformin" son el mismo término — el índice es case-insensitive por
construcción. Segundo, solo regresa el documento 0: coincide con todos los términos de la
query, mientras que los documentos 1 y 2 no comparten vocabulario con la query y `search`
solo devuelve hits con score positivo — una query sin solapamiento produce una lista
vacía, no filas con score cero.

Los dos perillas de BM25, `k1` (saturación de term-frequency, default 1.5) y `b`
(normalización de longitud, default 0.75), importan cuando los documentos varían en
longitud. El índice usa la variante de IDF de Lucene `ln(1 + (N - df + 0.5) / (df + 0.5))`,
que se mantiene positiva; los scores se redondean a cuatro decimales.

## Paso 3: Divide un documento largo en chunks para RAG

Los retrievers trabajan sobre chunks, no sobre documentos completos. Construye un
documento de política realista y divídelo con los defaults (64 palabras por chunk,
solapamiento de 16 palabras):

```python
from interview_prep.retrieval import chunk_text

long_doc = (
    "This clinical data policy defines how patient records flow through the analytics platform. "
    "All protected health information must be masked before leaving the secure enclave. "
    "Batch jobs run nightly and stream processing handles vitals telemetry in near real time. "
    "Every request is audited and retained for seven years under HIPAA rules. "
    "Analysts access de-identified extracts only, and re-identification attempts are blocked and logged. "
    "Third-party model endpoints never receive raw identifiers; the sanitizer rewrites them deterministically. "
    "Quarterly access reviews remove accounts that no longer require data access. "
    "Incidents must be reported to compliance within twenty four hours."
)

chunks = chunk_text(long_doc)
print(len(chunks))
print(len(chunks[0].split()))
print(chunks[0][:90] + " ...")
```

Salida esperada:

```text
2
64
This clinical data policy defines how patient records flow through the analytics platform. ...
```

Con `chunk_size=64` y `overlap=16` la ventana avanza 48 palabras por paso, así que la cola
del chunk 1 solapa la cabeza del chunk 2. Ese solapamiento es lo que te salva cuando un
hecho cruza una frontera — el clásico bug de retrieval "respuesta cortada a la mitad" que
los entrevistadores preguntan. Este documento de política tiene 96 palabras, así que
obtienes 2 chunks (palabras 0-63, luego 48-95); documentos más largos escalan el número de
chunks con el mismo paso de 48 palabras, y un documento de 64 palabras o menos se devuelve
como un único chunk.

## Paso 4: Simula un retriever denso

El retrieval híbrido fusiona una lista léxica (BM25) con una lista semántica (embeddings).
Este repo no tiene servicio de embeddings, así que simula el canal denso con un scorer de
solapamiento léxico — la *tubería* es idéntica a producción, donde sustituirías por FAISS
o un vector search administrado:

```python
from interview_prep.retrieval import tokenize

def overlap_score(query: str, doc: str) -> float:
    q, d = set(tokenize(query)), set(tokenize(doc))
    return len(q & d) / (len(q) * len(d)) ** 0.5

def dense_ranking(query: str, corpus: list[str], top_k: int = 3) -> list[int]:
    scored = sorted(
        ((i, overlap_score(query, doc)) for i, doc in enumerate(corpus)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [i for i, _ in scored[:top_k]]

query = "how do we rank documents with term frequency"
bm25_hits = index.search(query, top_k=3)
bm25_ranking = [doc_id for doc_id, _ in bm25_hits]
sim_ranking = dense_ranking(query, docs, top_k=3)
print("bm25 :", bm25_ranking)
print("dense:", sim_ranking)
```

Salida esperada:

```text
bm25 : [2, 4, 0]
dense: [2, 3, 4]
```

BM25 mete el documento 0 (su única coincidencia es la palabra "with") que el scorer de
solapamiento ignora, mientras que el scorer de solapamiento promueve el documento 3 por el
único término compartido "documents". Dos canales, dos opiniones. Ese desacuerdo es
exactamente lo que la fusión va a resolver.

## Paso 5: Fusiona los rankings con reciprocal rank fusion

```python
from interview_prep.retrieval import reciprocal_rank_fusion

def rrf_score(doc_id: int, rankings: list[list[int]], k: int = 60) -> float:
    return sum(
        1.0 / (k + rank)
        for ranking in rankings
        for rank, d in enumerate(ranking, 1)
        if d == doc_id
    )

rankings = [bm25_ranking, sim_ranking]
fused = reciprocal_rank_fusion(rankings, k=60)
print(fused)
for doc_id in fused:
    print(doc_id, round(rrf_score(doc_id, rankings), 4))
```

Salida esperada:

```text
[2, 4, 3, 0]
2 0.0328
4 0.032
3 0.0161
0 0.0159
```

RRF le da a cada documento `1 / (k + rank)` por cada lista en la que aparece, con `k=60`
amortiguando la cabeza de cada lista para que ningún canal domine solo. Lee la salida como
una historia: el documento 2 lidera ambas listas y gana de calle; el documento 3, que el
top-3 de BM25 *se perdió por completo*, aún queda tercero porque el canal denso lo rankeó
segundo. Falla una vez, rankea una vez, recupérate — ese es todo el argumento del retrieval
híbrido, y es la frase que decir en entrevistas.

## Paso 6: Evalúa respuestas estilo RAGAS

Ahora cierra el loop. Recupera contexto, genera una respuesta (pregrabada aquí — en
producción sustituye por tu llamada al LLM), y puntúala con `evaluate_rag_response`, que
calcula faithfulness estilo RAGAS (¿la respuesta está fundamentada en el contexto?) y
answer relevance (¿atiende la query?):

```python
from interview_prep.evaluator import evaluate_rag_response

q = "What is the recommended dosage of metformin for adults?"
ctx = [
    "The recommended starting dosage of metformin for adults is 500 mg twice daily with meals.",
    "Maximum daily dose should not exceed 2000 mg unless supervised by an endocrinologist.",
]

good = "The recommended starting dosage of metformin for adults is 500 mg twice daily with meals, and the maximum daily dose is 2000 mg."
hallucinated = "Metformin was discovered in 1922 and is derived from the French lilac plant, famous in herbal medicine."

print(evaluate_rag_response(q, good, ctx))
print(evaluate_rag_response(q, hallucinated, ctx))
print(evaluate_rag_response(q, good, []))
```

Salida esperada:

```text
{'faithfulness': 1.0, 'answer_relevance': 1.0, 'status': 'PASSED', 'reason': 'Metrics computed successfully.'}
{'faithfulness': 0.28, 'answer_relevance': 0.5, 'status': 'REVIEW_REQUIRED', 'reason': 'Metrics computed successfully.'}
{'faithfulness': 0.0, 'answer_relevance': 0.0, 'status': 'FAILED', 'reason': 'Empty context provided.'}
```

Las tres filas son los tres estados que le importan a tu puerta de CI:

- Respuesta fundamentada -> `PASSED` (ambas métricas en 1.0).
- Respuesta fluida pero sin fundamentos -> `REVIEW_REQUIRED` (faithfulness 0.28: la
  mayoría de sus palabras no está soportada por el contexto). Así se ve la alucinación
  para un juez automatizado.
- El retrieval no devolvió nada -> `FAILED` con una razón. Nunca evalúes una respuesta
  contra contexto vacío; falla el pipeline antes de la llamada al LLM.

## Paso 7: Convierte la evaluación en una puerta de regresión

Un score RAGAS que miras una vez es un demo; una puerta que bloquea despliegues es
LLMOps. Envuelve un golden set en un loop de aserciones:

```python
from interview_prep.evaluator import evaluate_rag_response

GOLDEN_SET = [
    {
        "query": q,
        "context": ctx,
        "response": good,
        "min_faithfulness": 0.8,
    },
    {
        "query": "How does BM25 rank documents?",
        "context": ["BM25 ranks documents using term frequency, inverse document frequency and length normalization."],
        "response": "BM25 ranks documents with term frequency, IDF, and document length normalization.",
        "min_faithfulness": 0.8,
    },
]

def run_gate(golden_set: list[dict]) -> bool:
    all_passed = True
    for case in golden_set:
        report = evaluate_rag_response(case["query"], case["response"], case["context"])
        ok = report["status"] == "PASSED" and report["faithfulness"] >= case["min_faithfulness"]
        print(case["query"][:40], report["status"], report["faithfulness"], "OK" if ok else "REGRESSION")
        all_passed = all_passed and ok
    return all_passed

print("gate:", "GREEN" if run_gate(GOLDEN_SET) else "RED")
```

Salida esperada:

```text
What is the recommended dosage of metf PASSED 1.0 OK
How does BM25 rank documents? PASSED 1.0 OK
gate: GREEN
```

Conecta `run_gate` en CI después de tu suite de tests: si devuelve `False`, el deploy se
detiene. Esa única función es la diferencia entre "evaluamos a veces" y "la evaluación es
una puerta de release" — di exactamente eso en la entrevista.

## Paso 8: Observabilidad de tokens y costos

Las evaluaciones vigilan la calidad; LLMOps también vigila el dinero. Estima tokens y
proyecta la factura mensual con los helpers de costo del repo (usa una heurística de
chars/4 por token):

```python
from interview_prep.cost import estimate_tokens, estimate_cost, monthly_projection

answer = "Escalate to the on-call endocrinologist and recheck glucose q15 minutes."
tokens_in = estimate_tokens("System: you are a clinical assistant. Answer from given context.")
tokens_out = estimate_tokens(answer)
print(tokens_in, tokens_out)

cost = estimate_cost(
    tokens_in=tokens_in,
    tokens_out=tokens_out,
    price_in_per_mtok=1.50,
    price_out_per_mtok=6.00,
)
print(cost)
print(monthly_projection(requests_per_day=25000, cost_per_request=cost))
```

Salida esperada (ambos strings son múltiplos exactos de cuatro caracteres, así que la
heurística de 4 caracteres por token es inequívoca):

```text
16 18
0.000132
99.0
```

`estimate_cost` convierte tokens a dólares con precios por millón de tokens
(`tokens / 1e6 * price`), y `monthly_projection` multiplica el volumen diario por el costo
por request por 30 días. La economía unitaria aquí es minúscula por request; el movimiento
de entrevista es escalarla — a 25,000 requests por día hasta los requests baratos se
componen, y por eso el costo-por-request pertenece al mismo dashboard que el faithfulness.

## Lo que aprendiste

- `tokenize` es la capa de vocabulario compartida: minúsculas, sin puntuación,
  determinista.
- `BM25Index` rankea por term frequency e IDF con perillas `k1`/`b` y devuelve tuplas
  `(doc_id, score)`, incluyendo hits de score cero.
- `chunk_text` construye ventanas con solapamiento (default 64/16, stride = size -
  overlap) para que los hechos que cruzan fronteras sobrevivan al retrieval.
- `reciprocal_rank_fusion` mergea listas rankeadas con `1/(k+rank)` para que un documento
  perdido por un canal aún pueda emerger por el otro.
- `evaluate_rag_response` devuelve el trío de veredictos estilo RAGAS sobre el que pones
  puertas de deploy: faithfulness, answer_relevance, status.
- `estimate_tokens` / `estimate_cost` / `monthly_projection` cierran el loop de LLMOps con
  economía unitaria.

## Ejercicios

### Ejercicio 1: Predice y verifica BM25

Antes de ejecutar nada, predice el documento top para la query `"hypertension blood
pressure medication"`. Luego corre `index.search` y compruébalo. ¿Qué documento gana y por
qué el documento 0 puntúa cero?

<details>
<summary>Solución</summary>

```python
print(index.search("hypertension blood pressure medication", top_k=3))
```

Salida:

```text
[(1, 5.1634), (0, 0.8144)]
```

El documento 1 contiene los tres términos distintivos ("hypertension", "blood pressure",
"medication"), así que domina. El documento 0 solo comparte la palabra "medication" y
puntúa un lejano 0.8144, mientras que el documento 2 no comparte nada y queda filtrado por
completo. El retrieval léxico es exact-match: la paráfrasis "high blood pressure"
puntuaría el documento 1 en cero y devolvería una lista vacía. Ese punto ciego es la
motivación del canal denso del Paso 4.
</details>

### Ejercicio 2: Sensibilidad de RRF a k

Re-fusiona los dos rankings del Paso 5 con `k=1` en lugar de `k=60` y compara el orden
fusionado. ¿Qué cambia y qué te dice eso sobre el parámetro de amortiguación?

<details>
<summary>Solución</summary>

```python
fused_k1 = reciprocal_rank_fusion([bm25_ranking, sim_ranking], k=1)
print("k=60:", reciprocal_rank_fusion([bm25_ranking, sim_ranking], k=60))
print("k=1 :", fused_k1)
```

Salida:

```text
k=60: [2, 4, 3, 0]
k=1 : [2, 4, 3, 0]
```

Para estas dos listas cortas el conjunto de ganadores es estable en ambos ajustes — un
buen sanity check — pero los *márgenes* difieren: con `k=1`, el rank-1 (1/2) empequeñece
al rank-3 (1/4), así que la cabeza de una sola lista domina más fácil. Con `k=60`, las
contribuciones se comprimen hacia `1/60`, y aparecer en *ambas* listas importa más que ser
primero en una. Ajusta `k` con queries etiquetadas, no por intuición.
</details>

### Ejercicio 3: Rompe la puerta a propósito

Agrega la respuesta alucinada de metformina a `GOLDEN_SET` con
`min_faithfulness: 0.8` y confirma que la puerta se pone ROJA. Luego repara el caso
reemplazando la respuesta por una fundamentada en `ctx` y confirma que vuelve a VERDE.

<details>
<summary>Solución</summary>

```python
bad_case = {"query": q, "context": ctx, "response": hallucinated, "min_faithfulness": 0.8}
print("gate:", "GREEN" if run_gate(GOLDEN_SET + [bad_case]) else "RED")
```

Salida:

```text
What is the recommended dosage of metf PASSED 1.0 OK
How does BM25 rank documents? PASSED 1.0 OK
What is the recommended dosage of metf REVIEW_REQUIRED 0.28 REGRESSION
gate: RED
```

La puerta falla cerrada: una respuesta sin fundamentos es un bloqueador de deploy, no una
advertencia. Cambiar `bad_case["response"]` por el string fundamentado `good` restaura el
VERDE. En CI imprimirías el campo `reason` del caso que falla y el diff entre el contexto
esperado y el real para que el arreglo esté a un clic.
</details>

## Lente de entrevista

Este tutorial es la columna vertebral de LLMOps que los entrevistadores exploran en tres
movimientos. Movimiento uno: "camíname tu stack de retrieval" — responde con los Pasos
2-5, enfatizando *por qué* híbrido (la historia de recuperación del Paso 5) y dónde muerden
las fronteras de chunk (el solapamiento del Paso 3). Movimiento dos: "¿cómo sabes que RAG
funciona en producción?" — responde con la semántica de métricas del Paso 6 y nombra los
estados de fallo explícitamente (`REVIEW_REQUIRED` para alucinación, `FAILED` para
retrieval vacío). Movimiento tres: "¿cómo se mantiene funcionando?" — responde con la
puerta de golden set del Paso 7 y la línea de costo del Paso 8, y ofrece por iniciativa
propia que umbrales como `min_faithfulness` son artefactos versionados, no vibras. Si te
preguntan qué agrega RAGAS sobre mirar con el ojo, la respuesta de una línea es: convierte
la calidad de la respuesta en un número testeable por regresión.

## Errores comunes

- `ModuleNotFoundError: No module named 'interview_prep.retrieval'` — un checkout viejo.
  Haz pull de la rama más reciente y re-ejecuta `pip install -e ".[dev]"`.
- `ModuleNotFoundError: No module named 'interview_prep'` — corriste Python desde fuera
  del repositorio o te saltaste la instalación editable. Activa el venv y re-ejecuta el
  Paso 1.
- Correr `python` fuera del venv (Python del sistema sin el paquete). Prefiere
  `source .venv/bin/activate` y verifica con `which python` (debe apuntar a
  `.venv/bin/`).
- `ValueError: docs must be a non-empty list` al construir `BM25Index([])` — el índice se
  niega a corpus vacíos; verifica `len(docs) > 0` antes de indexar.
- `ValueError: overlap must be smaller than chunk_size` desde `chunk_text` — la ventana
  debe avanzar de verdad; mantén `overlap` estrictamente por debajo de `chunk_size`.
- Un `[]` vacío de `index.search` significa cero solapamiento léxico con todos los
  documentos — maneja el caso de lista vacía antes de construir la lista de contexto para
  el LLM.
- Colab: olvidar el `%cd` al repo antes de `!pip install -e .` — la instalación tiene
  éxito pero `interview_prep` no es importable desde el directorio de trabajo del
  notebook. Re-ejecuta primero la celda del `%cd`.
- En Colab, `%pip install -e .` solo dura la sesión; tras un restart del runtime,
  re-ejecuta la celda de setup antes de cualquier import.
