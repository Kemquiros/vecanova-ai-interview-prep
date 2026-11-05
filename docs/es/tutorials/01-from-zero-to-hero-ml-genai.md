# Tutorial 1: From Zero to Hero — Uniendo ML tradicional y GenAI

> Versión en español del documento original en inglés: [01-from-zero-to-hero-ml-genai.md](../01-from-zero-to-hero-ml-genai.md)

> Para quién es: ingenieros preparando entrevistas de AI/ML Senior que saben escribir Python
> pero nunca han conectado modelos predictivos clásicos (detección de anomalías, recomendadores)
> con un pipeline de IA generativa. Presupuesto de tiempo: 60 minutos.

## Lo que vas a construir

1. Un detector de anomalías calibrado sobre telemetría clínica (scoring estilo Isolation Forest).
2. Un recomendador de productos basado en embeddings usando similitud de coseno.
3. Un "payload de contexto ML" determinista que inyecta salidas del modelo en un prompt de LLM —
   el patrón exacto que los entrevistadores quieren que articules para sistemas GenAI de producción.
4. Un mapeo del mismo patrón a cinco industrias (Healthcare, Fintech, E-commerce,
   Logistics, SaaS).

Cada paso corre localmente con la librería estándar solamente — sin GPU, sin API keys.

## Paso 1: Prepara el entorno

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make test
```

Salida esperada (abreviada; tu contador de tests crece conforme llegan módulos nuevos):

```text
Successfully installed interview_prep-0.1.0 ...
...
tests/integration/test_pipeline.py .                                    [ 25%]
tests/unit/test_evaluator.py ...                                        [100%]
============================== 4 passed in 0.38s ===============================
```

El extra `.[dev]` instala pytest, pytest-cov, ruff y build encima de las dependencias de
runtime (pydantic, typer, rich). `pip install -e .` pone `interview_prep` en tu path en modo
editable, así que cada import de este tutorial resuelve a `src/interview_prep/`.

**Variante de Google Colab:** sube la carpeta del repositorio (o tu fork) a Drive, abre
`notebooks/ai_interview_masterclass.ipynb` en Colab, y en una celda nueva ejecuta:

```python
%cd /content/vecanova-ai-interview-prep
!pip install -e ".[dev]"
```

Colab no tiene venv que activar — la instalación editable en el entorno de Colab es todo lo
que necesitas. Vuelve a ejecutar la celda de instalación una vez por sesión (los discos de
Colab son efímeros).

## Paso 2: Corre el detector de anomalías sobre telemetría clínica

```python
from interview_prep.traditional_ml import MockAnomalyDetector

detector = MockAnomalyDetector(threshold=2.0)

stable_vitals = [98.0, 100.0, 99.0, 101.0]          # heart rate, bpm
telemetry     = [120.0, 122.0, 210.0, 119.0]        # one systolic spike

print(detector.compute_anomaly_score(stable_vitals))
print(detector.is_anomaly(stable_vitals))
print(detector.compute_anomaly_score(telemetry))
print(detector.is_anomaly(telemetry))
```

Salida esperada:

```text
1.34
False
1.73
False
```

`compute_anomaly_score` mide a cuántas desviaciones estándar está la lectura más extrema
respecto a la media (un proxy de z-score). El `210.0` contra `[120, 122, 119]` infla tanto
la media como la desviación, dejando el score en 1.73 — por debajo del umbral de 2.0. Esto es
un momento de enseñanza deliberado: un outlier solitario dentro de una ventana pequeña se
enmascara parcialmente a sí mismo.

## Paso 3: Calibra el umbral (y esquiva la trampa de la desigualdad estricta)

```python
spike = [72.0, 74.0, 73.0, 71.0, 180.0]             # resting HR + one 180 bpm event

print(detector.compute_anomaly_score(spike))         # score sits exactly ON 2.0
print(detector.is_anomaly(spike))                    # strict '>' means NOT flagged

calibrated = MockAnomalyDetector(threshold=1.5)
print(calibrated.is_anomaly(spike))
```

Salida esperada:

```text
2.0
False
True
```

Relee esa segunda línea: el score es igual al umbral y `is_anomaly` usa un `>` estricto,
así que nada se dispara. A los entrevistadores les encanta esta clase de bug. En producción
harías (a) saber tu comparador exactamente, y (b) elegir el umbral desde un set de validación
negociando precisión contra recall, no por folclore. Bajarlo a 1.5 marca el pico —
al costo de más falsos positivos que enrutarás a una cola de revisión humana.

## Paso 4: Recomienda con embeddings y similitud de coseno

```python
from interview_prep.traditional_ml import MockRecommenderSystem

catalog = {
    "insulin_pump":   [1.0, 0.2, 0.1],   # axis 0: diabetes-care affinity
    "glucose_meter":  [0.9, 0.4, 0.2],
    "bp_monitor":     [0.1, 0.1, 0.95],  # axis 2: cardiology affinity
    "ecg_patch":      [0.2, 0.9, 0.1],   # axis 1: monitoring affinity
}
reco = MockRecommenderSystem(catalog)

print(round(reco.cosine_similarity([1.0, 0.0, 0.0], [0.9, 0.4, 0.2]), 4))
print(reco.recommend([0.95, 0.3, 0.15], top_k=2))
```

Salida esperada:

```text
0.8955
[{'item_id': 'insulin_pump', 'score': 0.9929}, {'item_id': 'glucose_meter', 'score': 0.9926}]
```

`recommend` puntúa cada ítem del catálogo contra el embedding de la query y devuelve el top-k
ordenado por similitud de coseno. El vector de la query se inclina al eje de cuidado de
diabetes, así que la bomba y el glucómetro ganan. Cambia el catálogo por embeddings de SKUs
y este es el servicio de e-commerce "clientes como tú también compraron"; cámbialo por
embeddings de planes de cuidado y es soporte de decisiones clínicas. La misma matemática,
distinto régimen de gobernanza — que es exactamente la historia de cinco industrias que
contarás en las entrevistas.

## Paso 5: Puentea las salidas del modelo hacia un prompt de LLM

El patrón de producción: el ML clásico produce **contexto determinista y auditable**;
el LLM razona sobre él pero nunca inventa números. Construye el payload explícitamente:

```python
import json
from interview_prep.traditional_ml import MockAnomalyDetector

def build_patient_context(telemetry: list[float], age: int, systolic_bp: int) -> str:
    detector = MockAnomalyDetector(threshold=1.5)
    return json.dumps(
        {
            "risk_flag": detector.is_anomaly(telemetry),
            "anomaly_score": detector.compute_anomaly_score(telemetry),
            "age": age,
            "systolic_bp": systolic_bp,
        },
        indent=2,
    )

context_json = build_patient_context([118.0, 121.0, 119.0, 184.0], age=67, systolic_bp=182)
print(context_json)

prompt = f"""You are a clinical decision-support assistant.
Patient telemetry context (computed by classical ML, treat as ground truth):
{context_json}
Draft a two-sentence alert for the care team.
Do not invent clinical facts that are not in the context."""
print(prompt[:150] + " ...")
```

Salida esperada:

```text
{
  "risk_flag": true,
  "anomaly_score": 1.73,
  "age": 67,
  "systolic_bp": 182
}
You are a clinical decision-support assistant.
Patient telemetry context (computed by classical ML, treat as ground truth):
{
  "risk_flag": true,
 ...
```

Dos propiedades hacen esto de grado entrevista: los números del prompt son reproducibles
(misma telemetría, mismo JSON), y la instrucción "no inventes hechos" ata al generador al
contexto estructurado. Cuando el entrevistador pregunte "¿cómo evitas que el LLM alucine
signos vitales?", este límite de payload es tu respuesta.

## Paso 6: Mapea el patrón a las cinco industrias

El patrón puente es idéntico; solo cambian los features, la semántica del umbral y el régimen
de cumplimiento:

| Industria | Modelo clásico que alimenta el contexto | Trabajo del LLM | Lente de cumplimiento |
|---|---|---|---|
| Healthcare / HIPAA | Score de anomalía de signos vitales (este tutorial) | Redactar alertas para el equipo de cuidado | PHI nunca sale del VPC; auditar cada prompt |
| Fintech / PCI-DSS | Score de fraude de transacciones (XGBoost) | Explicar transacciones marcadas | Sin datos de tarjetahabientes en los prompts |
| E-commerce | Scores del recomendador (este tutorial) | Personalizar copy de marketing | Sin PII en los event streams |
| Logistics | Score de riesgo de ETA / mantenimiento predictivo | Resumir planes ante disrupciones | Reglas de residencia de telemetría |
| SaaS LLMOps | Score de anomalía de uso multi-tenant | Redactar resúmenes de incidentes | Aislamiento de tenants en el retrieval |

## Lo que aprendiste

- `MockAnomalyDetector` puntúa la desviación de la media en desviaciones estándar y marca
  con un `>` estricto — un score igual al umbral NO se dispara.
- Los umbrales son una decisión de precisión/recall tomada sobre datos de validación, no un
  default.
- `MockRecommenderSystem` convierte embeddings en salida top-k ordenada y explicable vía
  similitud de coseno.
- El puente de producción es un payload de contexto JSON determinista inyectado en el
  prompt, con una restricción explícita de "no inventes hechos".

## Ejercicios

### Ejercicio 1: Barrido de umbrales

Escribe una función que reporte el veredicto del detector para los umbrales 1.0, 1.5, 2.0 y
2.5 sobre la muestra de pico `[72.0, 74.0, 73.0, 71.0, 180.0]`. ¿Qué umbrales se disparan?

<details>
<summary>Solución</summary>

```python
from interview_prep.traditional_ml import MockAnomalyDetector

spike = [72.0, 74.0, 73.0, 71.0, 180.0]
for t in (1.0, 1.5, 2.0, 2.5):
    verdict = MockAnomalyDetector(threshold=t).is_anomaly(spike)
    print(t, verdict)
```

Salida:

```text
1.0 True
1.5 True
2.0 False
2.5 False
```

El score es 2.0, así que todo umbral estrictamente por debajo de 2.0 se dispara y todo
umbral igual o superior queda en silencio — la frontera de desigualdad estricta en acción.
</details>

### Ejercicio 2: Sanity check de recomendación de e-commerce

Usando el catálogo del Paso 4, verifica que un embedding de query sesgado hacia el eje de
cardiología rankea `bp_monitor` primero, e imprime la lista completa ordenada con scores.

<details>
<summary>Solución</summary>

```python
from interview_prep.traditional_ml import MockRecommenderSystem

catalog = {
    "insulin_pump":  [1.0, 0.2, 0.1],
    "glucose_meter": [0.9, 0.4, 0.2],
    "bp_monitor":    [0.1, 0.1, 0.95],
    "ecg_patch":     [0.2, 0.9, 0.1],
}
reco = MockRecommenderSystem(catalog)
ranked = reco.recommend([0.15, 0.2, 0.9], top_k=4)
for item in ranked:
    print(item)
```

Salida:

```text
{'item_id': 'bp_monitor', 'score': 0.992}
{'item_id': 'glucose_meter', 'score': 0.4208}
{'item_id': 'ecg_patch', 'score': 0.3463}
{'item_id': 'insulin_pump', 'score': 0.2925}
```

La query sesgada a cardiología voltea el ranking de punta a punta y la lista regresa
ordenada por score descendente — una buena demo de geometría de embeddings para
entrevistadores que preguntan "¿por qué no solo keyword-match de productos?"
</details>

### Ejercicio 3: Sabor fintech — triage de fraude

Un servicio de pagos produce features horarias de transacciones
`[180.0, 45.0, 52.0, 61.0, 58.0]` (montos en USD). Decide si un agente de fraude debe ser
despertado, usando un umbral de 1.5, e imprime el payload de decisión que reenviaría.

<details>
<summary>Solución</summary>

```python
from interview_prep.traditional_ml import MockAnomalyDetector
import json

detector = MockAnomalyDetector(threshold=1.5)
features = [180.0, 45.0, 52.0, 61.0, 58.0]
score = detector.compute_anomaly_score(features)
payload = {"fraud_flag": detector.is_anomaly(features), "anomaly_score": score}
print(json.dumps(payload))
```

Salida:

```text
{"fraud_flag": true, "anomaly_score": 1.99}
```

La transacción de $180 está a 1.99 desviaciones estándar de la media de la hora, así que el
agente recibe el page con un payload estructurado — el mismo patrón puente del Paso 5, bajo
lenguaje PCI-DSS ("sin PAN en el payload", solo features agregados).
</details>

## Lente de entrevista

Los entrevistadores que exploran "ML clásico vs GenAI" están probando si sabes dónde
pertenece cada uno. El encuadre ganador: los modelos clásicos son dueños de la **evidencia
numérica determinista** (scores, rankings, probabilidades); el LLM es dueño del **lenguaje
sobre esa evidencia**. Defiende la frontera con las tres propiedades del payload del Paso 5 —
determinista, auditable, acotado — y ofrece el modo de fallo por iniciativa propia (un LLM
al que le pides calcular un z-score eventualmente se equivocará; un Isolation Forest no). Si
te piden un ejemplo concreto, camina los Pasos 2-5 de este tutorial de punta a punta; toma
noventa segundos y prueba que de hecho construiste el patrón.

## Errores comunes

- `ModuleNotFoundError: No module named 'interview_prep'` — no estás en la raíz del repo o
  te saltaste `pip install -e ".[dev]"`. Re-ejecuta el Paso 1 desde la raíz del repositorio.
- `ImportError: cannot import name 'MockRecommenderSystem'` — un typo en el nombre de la
  clase o una copia vieja de `build/lib` haciendo sombra a `src/`. Importa exactamente
  `from interview_prep.traditional_ml import MockRecommenderSystem`, y re-ejecuta
  `pip install -e ".[dev]"` si existe un directorio `build/` de un run de empaquetado viejo.
- `AttributeError: 'list' object has no attribute ...` — pasaste un dict donde se espera un
  `list[float]` (o viceversa). Ambas clases anotan los tipos de sus entradas; iguala las
  firmas del contrato.
- Usar un Python del sistema anterior a 3.10 — la base de código usa genéricos built-in
  `list[float]` (`requires-python = ">=3.10"`). Verifica con `python --version`; recrea el
  venv con un intérprete más nuevo si es necesario.
- `compute_anomaly_score([])` devuelve `0.0` por contrato — una ventana vacía no es una
  anomalía, son datos faltantes. En producción alertarías sobre *no hay datos* por separado.
- `cosine_similarity` con un vector cero devuelve `0.0` (no es error). Protege tus embeddings
  antes de confiar en el ranking.
