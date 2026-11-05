# Teoría 6: Arquitecturas de IA multi-industria y adaptación de dominio

> Versión en español del documento original en inglés: [06-multi-industry-ai-architectures.md](../../theory/06-multi-industry-ai-architectures.md)

## Respuestas relámpago para la entrevista

1. "El patrón de arquitectura se transfiere; lo que cambia son las restricciones — latencia, cumplimiento y velocidad de datos son la verdadera pregunta de entrevista."
2. "Fintech compra latencia con modelos más simples: una regresión logística calibrada que puntúa en 5ms le gana a una red profunda que puntúa en 90ms cuando el presupuesto de autorización es de 100ms."
3. "En salud fallo cerrado (fail closed); en e-commerce fallo abierto (fail open) con un fallback de popularidad — la clase de cumplimiento dicta la estrategia de fallo, no el gusto del ingeniero."
4. "PSI por debajo de 0.1 es estable, 0.1-0.25 significa investigar, por encima de 0.25 significa reentrenar o hacer rollback — la misma puerta de drift protege modelos de fraude, pronósticos de demanda y calidad de retrieval."
5. "El aislamiento multi-tenant es primero un problema del plano de datos: claves por namespace, cuotas por tenant y atribución de costos desde el día uno — meter tenant_id retroactivamente en un año de trazas es un proyecto, no un parche."

## 1. El framework de adaptación

Las entrevistas senior prueban si puedes llevar una arquitectura a un nuevo régimen de restricciones. Interioriza esta tabla — es la respuesta de mayor densidad en entrevistas multi-industria:

| Industria | Restricción dura | Patrón por defecto | Estrategia de fallo | Métricas principales |
|---|---|---|---|---|
| Healthcare | Cumplimiento + auditoría | RAG + guardrails + VPC privado | Fail closed | PII recall >= 0.99, faithfulness >= 0.95, p95 < 2.5s |
| Fintech | Latencia (p99 < 50ms scoring) | ML tabular en tiempo real + features de grafo | Fail open a motor de reglas | PR-AUC, tasa de FP, ms de scoring p99 |
| E-commerce | Escala + picos (5-10x) | Two-tower retrieval + ranker + bandits | Fail open a popularidad | Recall@10, NDCG@10, CVR, p95 < 150ms |
| Logistics | No estacionariedad | Gradient boosting + TFT + simulación | Fallback a baseline ingenuo | WAPE, MASE, bias |
| SaaS | Multi-tenancy + costo unitario | RAG + LLMOps + caché semántico | Degradar por tier de tenant | TTFT p95 < 800ms, costo/tenant, tasa de alucinación |

## 2. Profundidad por industria

### 2.1 Healthcare (HIPAA / PHI)

- **Restricción**: cero fuga de PHI, auditabilidad (logs de 6 años), riesgo de alucinación en soporte de decisiones clínicas.
- **Patrones**: anonimización en pre-pipeline (NER con Presidio + regex, puerta fail-closed), retrieval híbrido sobre guías clínicas, RAGAS faithfulness como puerta de lanzamiento y de CI, despliegue en VPC privado (SageMaker/Vertex) o API con BAA y zero-retention.
- **Números para citar**: PII recall >= 0.99 con detector en capas + re-escaneo de salida; puerta de faithfulness en 0.95; p95 < 2.5s end-to-end con streaming; trazas de auditoría del 100% de accesos a PHI.
- **Ángulo de entrevista**: tratamiento completo en `docs/theory/04-healthcare-ai-security-and-guardrails.md` — anonimización vs pseudonimización vs tokenización, PSI sobre distribuciones de entidades del sanitizador, war stories (fuga cross-patient por caché semántico, bug de RBAC en unmask).

### 2.2 Fintech (PCI-DSS / fraude)

- **Restricción**: el flujo de autorización te da un presupuesto total de ~100ms para la decisión de riesgo (sin contar saltos de red); el scoring del modelo debe ser p99 < 50ms, lookup de features < 5ms, reglas < 2ms.
- **Matemática del desbalance de clases**: los positivos de fraude rondan 0.1-0.5% — la accuracy es insignificante (predecir todo negativo da 99.5% de accuracy). Optimiza **PR-AUC**, no ROC-AUC; maneja el desbalance con `scale_pos_weight` (LightGBM/XGBoost) o focal loss; **calibra** probabilidades (isotónica > Platt para datos grandes) para que el umbral de operación sea una decisión de negocio, no un artefacto del modelo.
- **Economía del umbral**: el corte óptimo iguala el costo marginal de FN (dólares de fraude esperados) con el costo marginal de FP (labor de revisión + fricción del cliente). Cita el marco: "el umbral es una ecuación de costos; la recalculo cuando cualquiera de los dos lados cambia."
- **Features de grafo**: GNNs / detección de comunidades sobre el grafo de transacciones para AML (el lavado de dinero es una propiedad del grafo, no de la transacción) — pero servidas como *features* hacia un modelo rápido, no como inferencia GNN crítica para p99.
- **PCI-DSS**: nunca guardar PAN en texto libre; tokenización antes de que nada entre a analítica o al contexto del prompt; segmentar el entorno de datos de tarjetahabientes (CDE) de la infraestructura de entrenamiento.
- **Explainability**: valores SHAP por decisión alimentan las explicaciones de adverse action (ECOA/Reg B para crédito) — es un requisito legal, no un nice-to-have.
- **War story para contar**: drift del feature de velocidad tras días festivos, PSI 0.38 en velocity, FPR de 2.1% -> 7.4% en cuatro días; fallback interino al champion anterior, puerta semanal de PSI + promoción champion/challenger; FPR de vuelta a 2.3%.

### 2.3 E-commerce (Recomendación / Búsqueda)

- **Arquitectura**: two-tower retrieval (torre de usuario x torre de ítem, ANN top-1000) -> ranker con features cruzadas (top-20) -> capa de exploración con bandits. p95 < 150ms total; etapa de retrieval < 40ms.
- **Matemática de capacidad**: 10M ítems x 768 dims x 4B = 30.7GB fp32 crudo (~46-61GB con overhead de HNSW) -> sharding desde el día uno; fp16 lo reduce a la mitad. Los picos estacionales son de 5-10x — las réplicas de retrieval hacen autoscale; el ranker es lo que sometes a load testing.
- **Cold start**: embeddings de fallback basados en contenido + prior de popularidad con suavizado bayesiano (Beta(1,1) -> actualización alfa/beta); nunca almacenar un cero en frío.
- **Bandits**: Thompson sampling con 5-10% de tráfico de exploración y un piso duro de revenue — cita "métricas guardrail: la exploración no puede costar más de 0.5% de CVR durante la ventana de aprendizaje."
- **Métricas**: recall@10 (retrieval), NDCG@10 (ranking), CTR/CVR (online), con ponderación por propensión inversa para el sesgo de posición.
- **War story para contar**: un refresh de catálogo (cambios de precio/descripción) invalidó silenciosamente los embeddings de ítems — recall@10 cayó de 0.82 a 0.64; arreglado con embeddings versionados por content-hash + pipeline incremental de re-embed; recall recuperado a 0.84 tras cambiar también a un encoder más fuerte, todo verificado con dual-read en sombra.

### 2.4 Logistics (Pronóstico / Enrutamiento)

- **Restricción**: las series de demanda son no estacionarias (promos, clima, festivos), intermitentes para productos de baja rotación y jerárquicas (SKU x DC x región).
- **Modelado**: gradient-boosted trees con features de lags/rolling siguen siendo el baseline fuerte; Temporal Fusion Transformers cuando tienes covariables estáticas (atributos de almacén) + entradas conocidas del futuro (calendario de promos, festivos) + series observadas (clima); Prophet para baselines interpretables por serie. Concilia pronósticos jerárquicos (MinT o bottom-up para series dispersas).
- **Métricas**: **WAPE < 15% para productos de alta rotación**; MAPE explota en series de demanda casi nula — usa **MASE** (escalado contra un pronóstico naive estacional) para demanda intermitente. Bias por separado de varianza: un pronóstico con bias 0 que se equivoca en ambas direcciones es más sano que uno con bias de 2%.
- **Optimización**: OR-Tools para VRP; RL para enrutamiento dinámico solo detrás de un **digital twin validado contra 90 días de trazas GPS** — primero la brecha sim-to-real, después la política.
- **Clase de latencia**: esto es batch (pronósticos horarios/diarios) más un servicio de enrutamiento casi en tiempo real (p95 < 1s) — la industria más lenta de las cinco, y las entrevistas prueban si sobre-ingenierizas una latencia que el dominio no necesita.
- **War story para contar**: el pedido era pronosticar "todo"; se negoció un piloto de 3 SKUs con una puerta de WAPE; el piloto logró 9% de WAPE en productos de alta rotación y luego escaló al 40% del catálogo — la negociación de alcance es la habilidad senior aquí.

### 2.5 SaaS (LLMOps multi-tenant)

- **Aislamiento**: namespaces de vectores por tenant (namespaces de Pinecone / particiones de Milvus), tenant en *todas* las claves de caché, row-level security en Postgres, tests de acceso cross-tenant en CI tratados como tests de autenticación.
- **Vecinos ruidosos**: token buckets y cuotas por tenant; la ingesta masiva va a un carril dedicado para que un tenant enterprise no sature el endpoint de embeddings; colas de prioridad por tier de contrato.
- **Atribución de costos**: etiqueta cada traza con tenant_id; el costo por tenant debe ser una consulta de dashboard, no un proyecto forense. La caché semántica (scoped por tenant, similitud >= 0.97) recorta 25-40% del gasto de LLM en tráfico tipo FAQ.
- **Latencia**: TTFT p95 < 800ms con streaming; la concurrencia es un problema de memoria de KV-cache, no solo de pesos — presupuesta ~0.5GB de KV por stream de contexto 4k para modelos clase 7B, así que 36 streams concurrentes necesitan ~18GB de headroom de KV además de los pesos.
- **Eval por tier**: los contratos enterprise llevan cada más un SLA de faithfulness — haz que la puerta de eval sea parte de la revisión del contrato, no una ocurrencia tardía.

## 3. Tabla de trade-offs 1: VPC privado vs proveedores de API

La pregunta de despliegue en toda entrevista de industria regulada:

| Dimensión | VPC privado (SageMaker/Vertex, open weights) | Proveedor de API (con BAA/DPA) |
|---|---|---|
| Time to market | Semanas-meses | Horas-días |
| Costo a 10B tok/mes estables | ~$0.5-1.5 por 1M efectivo a 60%+ de utilización de GPU | $3-15 por 1M precio de lista |
| Control de datos | Total, apto para auditoría | Contractual; riesgo de cadena de subprocessors |
| Control de latencia | Autoscaling propio; la varianza de p99 es tuya | Efectos de cola del proveedor en picos |
| Carga operativa | Alta: pesos, tuning de KV-cache, GPUs | Baja: configuración y llaves |
| Úsalo cuando | Cargas PHI/PCI estrictas; > 2B tok/mes en estado estable | Tiers no regulados, prototipos, tráfico con picos |

Línea de decisión para entregar: "Hago prototipos con APIs y modelo el cruce: si el volumen en estado estable y la clase de cumplimiento apuntan al mismo lado, gana el VPC; si solo el cumplimiento apunta ahí, VPC *para ese tier* con un router compartido — el despliegue es por clase de dato, no por empresa."

## 4. Tabla de trade-offs 2: evaluación en streaming vs batch (transversal)

| Dimensión | Eval inline (streaming) | Eval batch (offline) |
|---|---|---|
| Fraude | Reglas + checks de calibración inline (la decisión ES la eval) | Revisión semanal de PR-AUC + drift |
| Healthcare | Re-escaneo de PII + proxy de faithfulness inline | RAGAS + trazas de auditoría humana para cumplimiento |
| E-commerce | Checks guardrail del bandit inline | Replay offline de NDCG/recall sobre datos logueados |
| SaaS | TTFT + checks de refusal inline | Dashboards de calidad por tenant, puertas de regresión |
| Logistics | n/a (dominio batch) | Backtest + PSI sobre residuales |

Línea para entregar: "La eval inline compra *poder de bloqueo* a costa de latencia; la eval batch compra *profundidad* a costa de demora de detección. Los sistemas regulados necesitan ambas: bloquear con rails determinísticos baratos, puntuar profundo en batch y alertar sobre la tendencia del batch."

## 5. Tabla de trade-offs 3: Prompt+RAG vs Fine-tune

| Dimensión | Prompt + RAG | Fine-tune |
|---|---|---|
| Tiempo de entrega | Días | 2-6 semanas |
| Costo incremental | Solo inferencia | $2k-50k por corrida + superficie de MLOps |
| Requisito de datos | Nada por adelantado | 1k-100k ejemplos etiquetados |
| Frescura de conocimiento | Refresh del índice (minutos) | Congelado al momento del entrenamiento |
| Mejor para | Tareas de conocimiento, citas, contenido cambiante | Cumplimiento de formato, estilo, destilación a un modelo más pequeño/rápido, confiabilidad de structured output |

Regla: "El fine-tuning es el último paso, no el primero — prompt, tools, orquestación, RAG, y solo entonces fine-tune, cuando puedo nombrar la métrica que mueve y que la pila de abajo no puede mover."

## 6. La puerta de drift transversal (una implementación, cinco consumidores)

El drift es el enemigo compartido. El repo lo centraliza en `src/interview_prep/drift.py` — PSI con $B=10$ bins sobre los deciles de referencia:

$$\text{PSI} = \sum_{i=1}^{B} (A_i - E_i)\ln\frac{A_i}{E_i}$$

Umbrales: < 0.1 estable, 0.1-0.25 moderado (investigar), > 0.25 severo (reentrenar/rollback); el estadístico KS como complemento no paramétrico.

```python
from interview_prep.drift import (
    population_stability_index, ks_statistic, drift_verdict,
)

psi = population_stability_index(expected, actual, bins=10)
verdict = drift_verdict(psi)            # STABLE | MODERATE | SEVERE
ks = ks_statistic(expected, actual)     # sup |F_a - F_b|, p < 0.05 marca el shift
```

Qué le alimenta cada industria: fraude -> features de velocidad/monto/mix de comercios; e-commerce -> distribución de clústeres de embeddings de queries y baseline de CTR; logistics -> residuales de pronóstico por clase de serie; healthcare -> longitudes de notas y mix de códigos ICD-10; SaaS -> tópicos de query por tenant. La misma función, distintas listas de features — ese es el punto de la arquitectura.

## 7. Historias de guerra en producción y errores comunes

1. **Tormenta de falsos positivos de fraude tras una semana festiva** (PSI 0.38 en el feature de velocidad, FPR 2.1% -> 7.4% en cuatro días). Mitigación: puerta semanal de PSI sobre el top-20 de features, promoción champion/challenger con rollback automático, y un fallback a motor de reglas para que el camino de la transacción nunca dependa duramente de la disponibilidad del modelo de ML.
2. **Embeddings obsoletos tras un refresh de catálogo** (recall@10 de 0.82 a 0.64 en dos semanas). Mitigación: embeddings de ítems versionados por content-hash, re-embed incremental ante eventos de cambio, dual-read en sombra para verificar cambios de encoder antes del cutover.
3. **Los SKUs en promo rompieron el pronóstico de demanda** porque el calendario de promos no era covariable — WAPE de 21% en semanas de promo. Mitigación: covariables known-future (promos, festivos) obligatorias en el contrato de features; WAPE de 21% -> 11%; y una "revisión de pronóstico" que inspecciona específicamente las semanas de promo, no el agregado.
4. **La ingesta masiva de un tenant enterprise saturó el endpoint compartido de embeddings** y degradó el TTFT de todos los tenants. Mitigación: cuotas por tenant, carril de ingesta dedicado con backpressure, y SLOs de latencia scoped por tenant para que el vecino ruidoso sea medible y exigible.
5. **La política de enrutamiento con RL se veía brillante en simulación y falló el piloto** — el digital twin usaba tiempos de viaje optimistas. Mitigación: validar el simulador contra 90 días de trazas GPS primero (métrica de brecha sim-to-real sobre percentiles de tiempo de viaje); ninguna política se despliega hasta que el error de simulación en p90 de tiempos de viaje sea menor a 10%.
6. **La atribución de costos por tenant no existió el primer año** — nadie podía responder "¿cuánto nos cuesta este cliente?". Mitigación: tenant_id en cada traza desde el día uno en proyectos greenfield; para brownfield, un proyecto de backfill desde logs del proveedor con un nivel de confianza documentado. El costo por tenant es un insumo de economía unitaria, no un detalle operacional.

## 8. Lo que los entrevistadores realmente preguntan

**P1: "Has construido sistemas de IA en cinco industrias. ¿Qué cambia realmente entre ellas?"**
Respuesta modelo: entrega la tabla de la sección 1 textual, luego elige una fila y profundiza dos niveles (la economía de umbrales de fraude o el fail-closed de salud son las más fuertes). No narres las cinco filas — la tabla muestra amplitud; un solo deep dive muestra profundidad.

**P2: "Modelo de fraude vs recomendador — compara los presupuestos de latencia y las estrategias de fallo."**
Respuesta modelo: "Fraude: ~100ms en total, scoring p99 < 50ms, fail *open* a un motor de reglas porque bloquear pagos es peor que dejar pasar una transacción mala... salvo que la clase de fraude sea severa, en cuyo caso se falla a revisión manual. Recomendador: 150ms p95, fail *open* a popularidad porque un estante vacío pierde más que uno genérico. El mismo objetivo de disponibilidad, semánticas de fallo opuestas — porque el blast radius es distinto."

**P3: "¿Cómo detectas drift en un producto LLM?"**
Respuesta modelo: "Tres capas: PSI sobre distribuciones de entrada (longitudes de query, clústeres de embeddings, mixes de entidades) semanal; evals de regresión sobre un gold set estratificado de tráfico real; y proxies de resultado (tasa de refusal, tasa de escalación, muestra de faithfulness) monitoreados por hora. Las puertas de `drift.py` de este repo son la implementación de referencia — el arte está en elegir *qué* distribuciones binnear, no en el estadístico."

**P4: "¿Cuándo harías fine-tuning?"**
Respuesta modelo: la regla de la sección 5, luego un caso concreto: "Destilé una cadena RAG de 70B a un fine-tune de 8B para una tarea casi de clasificación cuando el p99 de latencia y el $/query lo exigían — la eval mostró que el modelo pequeño igualaba el 98% de las decisiones de la cadena, lo que pagó la corrida de entrenamiento en tres semanas."

**P5: "Diseña la atribución de costos para una plataforma LLM multi-tenant."**
Respuesta modelo: "Etiqueta en el borde: tenant_id en cada request, propagado por cada span. Tokens por tenant por modelo por feature se consolidan en un libro de costos; los hits de caché semántica y los descuentos de prompt-cache se acreditan de vuelta. Chargeback o showback por tier de contrato, con un piso por tenant para costos fijos. La restricción de diseño es que la atribución debe sobrevivir al camino de fallback — si lo sirve el proveedor secundario, el libro igual lo registra."

**P6: "0.3% de positivos de fraude — camíname tus decisiones de entrenamiento."**
Respuesta modelo: "PR-AUC como métrica principal, scale_pos_weight o focal loss para el desbalance, split temporal (nunca aleatorio — leakage), calibración de probabilidades para que el umbral sea un dial de negocio, y umbral fijado por la ecuación de costos. Reporta la precisión en el punto de operación y la carga de la cola de revisión, porque eso es lo que siente el equipo de operaciones de fraude."

**P7: "Lanza un feature LLM bajo HIPAA en seis semanas. Adelante."**
Respuesta modelo: "Semana 1: mapa de flujos de datos + decisiones de BAA, esqueleto del sanitizador fail-closed. Semanas 2-3: RAG sobre chunks desidentificados en un VPC, gold set + harness de eval. Semana 4: guardrails (input rails, re-escaneo de PII) + logging de auditoría. Semana 5: red team + baselines de PSI, shadow mode con clínicos. Semana 6: rollout por etapas con la puerta de faithfulness activa. La disciplina es que los ítems de cumplimiento están en el camino crítico desde el día 1 — el cumplimiento remendado es como consigues las war stories de la sección 7."

## 9. Posicionamiento para roles remotos LATAM -> US (bonus)

- **Benchmarks de compensación**: levels.fyi para anclas de banda; LinkedIn Salary para contexto regional. Cita rangos, no vibras.
- **Estructura**: la mayor parte del trabajo remoto LATAM-US corre por plataformas EOR/contractor (Deel, Remote) — conoce la forma fiscal/de cumplimiento a nivel conversacional; los hiring managers de US preguntan.
- **Expectativa de solapamiento**: 4+ horas de solapamiento con ET es el estándar de facto; enúncialo proactivamente.
- **Diferenciación**: war stories cuantificadas (el STAR bank de este repo), alfabetización en cumplimiento (vocabulario HIPAA/PCI usado correctamente) y documentación bilingüe — la mayoría de candidatos ofrece solo una de las tres.

Referencias: https://www.levels.fyi/ , https://www.linkedin.com/salary/ , https://www.deel.com/ , https://remote.com/

## 10. Lecturas recomendadas

- Documentación de SHAP: https://shap.readthedocs.io/en/latest/
- Temporal Fusion Transformers (paper): https://arxiv.org/abs/1912.09363
- Documentación de Prophet: https://facebook.github.io/prophet/
- Documentación de LightGBM (desbalance/`scale_pos_weight`): https://lightgbm.readthedocs.io/en/stable/
- Estándar PCI DSS v4.0: https://www.pcisecuritystandards.org/standards/pci-dss/
- Documentación de Pinecone (namespaces): https://docs.pinecone.io/
- Documentación de Milvus (particiones): https://milvus.io/docs
- Wiki de FAISS (selección de índice): https://github.com/facebookresearch/faiss/wiki
- Google OR-Tools (VRP/enrutamiento): https://developers.google.com/optimization
- OWASP Top 10 para Aplicaciones LLM: https://genai.owasp.org/

Referencias cruzadas: `src/interview_prep/drift.py` (`population_stability_index`, `ks_statistic`, `drift_verdict`), `src/interview_prep/cost.py` (`estimate_tokens`, `estimate_cost`, `monthly_projection`), `src/interview_prep/evaluator.py` (`evaluate_rag_response`).
