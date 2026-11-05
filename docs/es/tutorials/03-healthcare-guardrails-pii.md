# Tutorial 3: Guardrails de salud — sanitización de PII y detección de drift

> Versión en español del documento original en inglés: [03-healthcare-guardrails-pii.md](../03-healthcare-guardrails-pii.md)

> Para quién es: ingenieros preparando entrevistas de AI/ML Senior en industrias
> reguladas (Healthcare/HIPAA, Fintech/PCI-DSS). Construirás una capa de guardrail de PII,
> sondearás sus puntos ciegos con honestidad, pondrás una respuesta RAG detrás de ella, y
> detectarás drift estadístico en una distribución de features simulada antes de que
> degrade el modelo en silencio. Presupuesto de tiempo: 70 minutos.

## Lo que vas a construir

1. Una pasada de sanitización de PII sobre notas clínicas realistas usando `sanitize_pii`.
2. Una lista documentada de sus puntos ciegos (la honestidad que los entrevistadores
   premian).
3. Una puerta de respuesta RAG con guardia: sanitizar -> evaluar -> bloquear o liberar.
4. Un monitor de drift sobre una distribución de feature de laboratorio simulada usando
   PSI y el estadístico KS, cableado a veredictos STABLE / MODERATE / SEVERE y acciones
   concretas.

## Paso 1: Prepara el entorno

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -c "from interview_prep.evaluator import sanitize_pii; from interview_prep.drift import population_stability_index; print('imports ok')"
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

## Paso 2: Sanitiza una nota clínica

```python
from interview_prep.evaluator import sanitize_pii

note = "Patient John Doe (john.doe@hospital.org), phone +1 305 555 0142, SSN 545-12-8890."
print(sanitize_pii(note))

short_note = "Patient John Doe (john.doe@hospital.org)"
print(sanitize_pii(short_note))
```

Salida esperada:

```text
Patient John Doe ([EMAIL_REDACTED]), phone +[PHONE_REDACTED], SSN [SSN_REDACTED].
Patient John Doe ([EMAIL_REDACTED])
```

Emails, números de teléfono con código de país y SSNs se reescriben a placeholders
estables y determinísticos. El determinismo importa: la misma entrada siempre produce la
misma máscara, así que logs, evaluaciones y prompts se mantienen comparables entre
corridas. Nota la peculiaridad cosmética de la salida — el código de país `+1` queda
*fuera* del match del teléfono y sobrevive como `+[PHONE_REDACTED]`. Inofensivo aquí,
pero te dice que el regex espera un formato internacional.

## Paso 3: Sondea los puntos ciegos (no te saltes esto)

Un guardrail que no has intentado romper es una afirmación, no un control. Ataca tu
propio sanitizador:

```python
from interview_prep.evaluator import sanitize_pii

attempts = [
    "Call Maria at 305-555-0142 or maria.lopez@clinic.org about claim 2024-09-77.",
    "MRN 00432157, DOB 04/12/1958, room 7W-214.",
]
for attempt in attempts:
    print(sanitize_pii(attempt))
```

Salida esperada:

```text
Call Maria at 305-555-0142 or [EMAIL_REDACTED] about claim 2024-09-77.
MRN 00432157, DOB 04/12/1958, room 7W-214.
```

Ambas filas filtran PHI: el teléfono de diez dígitos sin código de país (la máscara
espera un prefijo de código de país), y el MRN, la fecha de nacimiento y el número de
habitación (no existen reglas para ellos). Esta es exactamente la conversación de HIPAA
Safe Harbor: una capa de regex es *un control*, no el programa completo. La producción la
apila con herramientas de reconocimiento de entidades (Presidio), redacción a nivel de
formato en la ingesta, y políticas de campos deny-by-default. Decir "sanitize_pii solo ya
nos hace cumplir HIPAA" es una bandera roja de entrevista; decir "es la primera capa
determinista de una pila de defensa en profundidad" es la respuesta senior.

## Paso 4: Pon la respuesta RAG detrás del sanitizador

La sanitización pertenece *antes* de que el payload llegue a cualquier modelo externo, y
la misma puerta debe re-chequear la respuesta generada. Cablea sanitizador más evaluador
en una sola puerta:

```python
from interview_prep.evaluator import sanitize_pii, evaluate_rag_response

def guarded_answer(query: str, response: str, context: list[str]) -> dict:
    safe_context = [sanitize_pii(c) for c in context]
    safe_response = sanitize_pii(response)
    pii_was_present = safe_response != response or any(
        safe_c != c for safe_c, c in zip(safe_context, context)
    )
    report = evaluate_rag_response(query, safe_response, safe_context)
    return {
        "blocked": report["status"] == "FAILED",
        "pii_was_present": pii_was_present,
        "report": report,
    }

query = "What is the recommended dosage of metformin for adults?"
context = [
    "The recommended starting dosage of metformin for adults is 500 mg twice daily. Contact Dr. Alice Kim (alice.kim@hospital.org) for exceptions.",
]
response = "Start metformin at 500 mg twice daily; contact alice.kim@hospital.org for exceptions."

print(guarded_answer(query, response, context))
```

Salida esperada:

```text
{'blocked': False, 'pii_was_present': True, 'report': {'faithfulness': 0.95, 'answer_relevance': 0.5, 'status': 'PASSED', 'reason': 'Metrics computed successfully.'}}
```

Lee el payload como lo leería un auditor: tanto el contexto recuperado como la respuesta
redactada contenían el email de un médico, la puerta enmascaró ambos determinísticamente
antes de evaluar (`pii_was_present: True`), y la respuesta siguió lo bastante fundamentada
para liberarse (`PASSED`). Si el retrieval alguna vez devuelve un contexto vacío,
`evaluate_rag_response` regresa `status="FAILED"` y `blocked` cambia a `True` — la puerta
falla cerrada, no abierta.

## Paso 5: Simula una distribución de feature y detecta drift

Los guardrails protegen el prompt; la detección de drift protege al *modelo*. Simula un
feature de laboratorio (glucosa en ayunas, mg/dL) como se vio en entrenamiento versus el
tráfico de esta semana:

```python
import random
from interview_prep.drift import population_stability_index, ks_statistic, drift_verdict

random.seed(42)  # reproducibility: same numbers every run
expected      = [random.gauss(120.0, 12.0) for _ in range(5000)]   # training distribution
actual_stable = [random.gauss(120.5, 12.2) for _ in range(5000)]   # this week: healthy
actual_shift  = [random.gauss(132.0, 12.0) for _ in range(5000)]   # upstream unit change

psi_stable = population_stability_index(expected, actual_stable)
psi_shift  = population_stability_index(expected, actual_shift)
print(psi_stable, drift_verdict(psi_stable))
print(psi_shift, drift_verdict(psi_shift))
print(ks_statistic(expected, actual_stable))
print(ks_statistic(expected, actual_shift))
```

Salida esperada (PSI usa bins de cuantiles tomados de la muestra esperada, con buckets
vacíos pisados en 1e-6, así que los decimales son reproducibles de corrida a corrida):

```text
0.003421 STABLE
0.971624 SEVERE
0.022
0.3914
```

El Population Stability Index binnear ambas distribuciones y acumula
`(actual% - expected%) * ln(actual% / expected%)`; por debajo de 0.10 es
convencionalmente estable, 0.10-0.25 moderado, por encima de 0.25 severo — que es
exactamente lo que `drift_verdict` codifica. La muestra estable puntúa 0.003421
(indistinguible de los datos de entrenamiento) mientras que un shift de media de una
desviación estándar (120 -> 132 mg/dL, por ejemplo, una recalibración del analizador de
laboratorio) puntúa 0.971624, en pleno SEVERE. El estadístico KS (la brecha más grande
entre las dos CDFs empíricas) lo corrobora: 0.022 para tráfico estable, 0.3914 para el
desplazado.

## Paso 6: Cablea los veredictos a acciones

Un veredicto sobre el que nadie actúa es teatro de telemetría. Vincula cada veredicto a
una respuesta operacional:

```python
from interview_prep.drift import population_stability_index, drift_verdict

VERDICT_ACTIONS = {
    "STABLE":   "log and continue",
    "MODERATE": "page on-call analyst; sample responses for manual review",
    "SEVERE":   "freeze model deploys; route traffic to fallback; open incident",
}

def drift_action(expected: list[float], actual: list[float]) -> str:
    psi = population_stability_index(expected, actual)
    verdict = drift_verdict(psi)
    return f"{verdict}: {VERDICT_ACTIONS[verdict]}"

print(drift_action(expected, actual_stable))
print(drift_action(expected, actual_shift))
```

Salida esperada:

```text
STABLE: log and continue
SEVERE: freeze model deploys; route traffic to fallback; open incident
```

En producción la rama SEVERE publica en tu canal de incidentes y voltea un feature flag
que re-enruta el detector de anomalías a la versión anterior del modelo — el camino de
rollback se define *antes* de que llegue el drift, nunca se improvisa durante.

## Lo que aprendiste

- `sanitize_pii` enmascara determinísticamente emails, teléfonos internacionales y SSNs —
  y se demuestra que pierde teléfonos de diez dígitos sin código de país y campos
  MRN/fecha de nacimiento, así que es la capa uno de una pila de defensa en profundidad,
  no un certificado de cumplimiento.
- La puerta de guardrail sanitiza contexto y respuesta *antes* de evaluar y falla cerrada
  cuando el retrieval no devuelve nada.
- `population_stability_index` más `drift_verdict` convierten un shift de distribución en
  un veredicto operacional (STABLE / MODERATE / SEVERE); `ks_statistic` es una segunda
  opinión independiente.
- Los veredictos se mapean a acciones pre-acordadas: log, page, freeze-and-rollback.

## Ejercicios

### Ejercicio 1: Endurece el sanitizador

Escribe `sanitize_bare_phone(text)` que también enmascare teléfonos de diez dígitos sin
código como `305-555-0142`, y verifícalo sobre el ejemplo que filtra del Paso 3.

<details>
<summary>Solución</summary>

```python
import re
from interview_prep.evaluator import sanitize_pii

def sanitize_bare_phone(text: str) -> str:
    text = sanitize_pii(text)
    return re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[PHONE_REDACTED]", text)

leaky = "Call Maria at 305-555-0142 or maria.lopez@clinic.org about claim 2024-09-77."
print(sanitize_bare_phone(leaky))
```

Salida:

```text
Call Maria at [PHONE_REDACTED] or [EMAIL_REDACTED] about claim 2024-09-77.
```

Apilar un patrón más estricto *después* de la pasada estándar es la forma correcta de
producción: funciones wrapper como esta son cómo extiendes un sanitizador compartido sin
editar código del que dependen otros equipos. (Extensión del ejercicio: envuelve MRNs
`\bMRN\s?\d{8}\b` y fechas de la misma manera.)
</details>

### Ejercicio 2: ¿Qué tan grande el shift antes de la alarma?

Barrer el shift de distribución de 0.0 a 1.25 desviaciones estándar en pasos de 0.25 e
imprimir PSI y veredicto para cada uno. ¿Dónde muerden los umbrales MODERATE y SEVERE?

<details>
<summary>Solución</summary>

```python
import random
from interview_prep.drift import population_stability_index, drift_verdict

random.seed(42)
expected = [random.gauss(120.0, 12.0) for _ in range(5000)]
for shift_sigma in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
    actual = [random.gauss(120.0 + shift_sigma * 12.0, 12.0) for _ in range(5000)]
    psi = population_stability_index(expected, actual)
    print(f"shift={shift_sigma:.2f} sigma  psi~{psi:.4f}  {drift_verdict(psi)}")
```

Patrón esperado (los veredictos son la propiedad robusta; los decimales cambian con el
tamaño de muestra):

```text
shift=0.00 sigma  psi~0.0032  STABLE
shift=0.25 sigma  psi~0.0721  STABLE
shift=0.50 sigma  psi~0.2404  MODERATE
shift=0.75 sigma  psi~0.6021  SEVERE
shift=1.00 sigma  psi~0.9063  SEVERE
shift=1.25 sigma  psi~1.4151  SEVERE
```

MODERATE abre justo alrededor de media sigma para este feature (0.2404, un pelo bajo el
límite de 0.25) y SEVERE se dispara a tres cuartos. Dos detalles que vale la pena decir
en voz alta en una entrevista: incluso la fila de shift cero no es exactamente cero (dos
muestras finitas de la misma distribución aún difieren levemente), y PSI crece
superlinealmente con el shift, así que una lectura "severe" te dice la *forma* de la
respuesta que necesitas (congelar e investigar), mientras que el barrido te dice *qué tan
temprano* se dispara tu alarma.
</details>

### Ejercicio 3: Combina ambos controles en una checklist de release

Escribe `release_check(context, response, expected_dist, actual_dist)` que devuelva una
única decisión GO / NO-GO que exija tanto que la puerta de PII pase como que el drift esté
STABLE.

<details>
<summary>Solución</summary>

```python
from interview_prep.evaluator import sanitize_pii, evaluate_rag_response
from interview_prep.drift import population_stability_index, drift_verdict
import random

random.seed(42)
expected = [random.gauss(120.0, 12.0) for _ in range(5000)]
actual_shift = [random.gauss(132.0, 12.0) for _ in range(5000)]

def release_check(query, context, response, expected_dist, actual_dist) -> dict:
    report = evaluate_rag_response(query, sanitize_pii(response),
                                   [sanitize_pii(c) for c in context])
    psi = population_stability_index(expected_dist, actual_dist)
    verdict = drift_verdict(psi)
    go = report["status"] == "PASSED" and verdict == "STABLE"
    return {"decision": "GO" if go else "NO-GO",
            "eval_status": report["status"], "drift": verdict, "psi": psi}

q = "What is the recommended dosage of metformin for adults?"
ctx = ["The recommended starting dosage of metformin for adults is 500 mg twice daily with meals."]
good = "Start metformin at 500 mg twice daily with meals."
print(release_check(q, ctx, good, expected, expected))
print(release_check(q, ctx, good, expected, actual_shift))
```

Salida:

```text
{'decision': 'GO', 'eval_status': 'PASSED', 'drift': 'STABLE', 'psi': 0.0}
{'decision': 'NO-GO', 'eval_status': 'PASSED', 'drift': 'SEVERE', 'psi': 0.897629}
```

La primera corrida pasa ambos controles (calidad `PASSED`, drift `STABLE`) -> `GO`. La
segunda falla solo por drift (`SEVERE`) -> `NO-GO`, aunque la calidad de la respuesta no
cambie — el comportamiento correcto, porque una distribución de feature desplazada
invalida las respuestas de mañana, no las de hoy. (El psi aquí es 0.897629 en lugar del
0.971624 del Paso 5 porque esta muestra de shift se extrajo tras un stream aleatorio más
corto; PSI es un estadístico de muestra, otra razón para fijar tus semillas en tests.) Dos
controles independientes combinados con un AND lógico es precisamente cómo debes
describir la gobernanza de release en una entrevista de industria regulada.
</details>

## Lente de entrevista

Las entrevistas de industria regulada prueban si tratas el cumplimiento como una propiedad
de ingeniería o como una diapositiva. Tres sondas y tus respuestas de este laboratorio:
(1) "¿Cómo mantienes la PHI fuera del modelo?" — la puerta del Paso 4, más la admisión del
Paso 3 de que los regex pierden MRNs y teléfonos sin código, más los compañeros de pila
nombrados (Presidio, redacción al momento de ingesta). (2) "Las métricas de un modelo
cayeron en silencio el trimestre pasado — ¿qué pasó?" — la historia del Paso 5: nadie tocó
el modelo, la *distribución de entrada* se movió una sigma; PSI lo atrapó, KS lo
corroboró, el veredicto impulsó un freeze-and-rollback. (3) "¿Quién es dueño del umbral?"
— las convenciones de PSI 0.10/0.25 son defaults de la industria, pero las calibras con
datos de validación y las re-revisas trimestralmente, porque un umbral es una decisión de
negocio codificada en código. Ofrecer el modo de fallo antes de que te pregunten — "los
sanitizadores son superficies de ataque; nosotros hacemos red team sobre el nuestro" — es
lo que separa las respuestas senior de las mid.

## Errores comunes

- `ModuleNotFoundError: No module named 'interview_prep.drift'` — checkout obsoleto; haz
  pull de la rama más reciente y re-ejecuta `pip install -e ".[dev]"`.
- `ModuleNotFoundError: No module named 'interview_prep'` — correr desde fuera del
  repositorio o saltarse la instalación editable; rehaz el Paso 1.
- Números de PSI distintos en cada corrida — te saltaste `random.seed(42)`; las
  simulaciones deben ser reproducibles o tus tests de drift no son testeables por
  regresión.
- Esperar que `305-555-0142` quede enmascarado — la regla de teléfono exige un prefijo de
  código de país; ver el Paso 3 y el Ejercicio 1.
- Esperar que MRN/fecha de nacimiento/número de habitación queden enmascarados — no hay
  reglas para ellos; ese es el punto ciego documentado, no un bug que reportar.
- `ZeroDivisionError` en un PSI hecho a mano — debes pisar las proporciones de bucket con
  un epsilon pequeño antes de tomar logaritmos; la implementación del contrato maneja
  buckets vacíos.
- PSI diminuto sobre datos evidentemente desplazados — comparaste solo unos cuantos
  cientos de muestras; PSI necesita suficientes observaciones por bucket (cientos a
  miles) para ser significativo.
- Colab: `%pip install -e .` dura solo la sesión — tras un restart del runtime, re-ejecuta
  la celda de setup antes de importar.
