# Teoría 1: Fundamentos avanzados de machine learning para AI Engineers

> Versión en español del documento original en inglés: [01-advanced-machine-learning-foundations.md](../../theory/01-advanced-machine-learning-foundations.md)

La mitad de machine learning clásico del loop senior: selección de métricas bajo restricciones
de negocio, interioridades de gradient boosting, detección de anomalías, sistemas de
recomendación, calibración, explicabilidad y drift. El enfoque es deliberadamente opinionado:
los entrevistadores de empresas de nivel FAANG y de consultoras globales no comprueban si
puedes recitar la fórmula del F1. Comprueban si puedes defender una elección de métrica frente
a un modelo de costos, enunciar cuándo cambiarías de opinión y conectar las matemáticas con un
sistema ya desplegado.

## Respuestas relámpago para la entrevista

- "ROC-AUC solo me dice que el modelo rankea bien. En fraude elijo el threshold a partir del
  presupuesto de capacidad de revisión —cuántos casos puede despachar el equipo de operaciones
  por día— y reporto la precisión en ese punto de operación, porque ese es el número que siente
  el negocio."
- "Trato la calibración como un deliverable de primera clase, no como un bonus: un modelo con
  0.92 de AUC y probabilidades mal calibradas igual rompe cada cálculo de costos aguas abajo que
  multiplica probabilidad por impacto."
- "Gradient boosting es descenso de gradiente funcional en el espacio de funciones: cada árbol
  ajusta el gradiente negativo de la pérdida, y la contribución de XGBoost es una expansión de
  Taylor de segundo orden de esa pérdida más regularización explícita, lo que deja los pesos de
  las hojas en forma cerrada."
- "Isolation Forest funciona porque las anomalías son 'pocas y diferentes': un split aleatorio
  paralelo a los ejes las aísla en menos cortes, así que la longitud de camino esperada se
  convierte directamente en un score de anomalía entre 0 y 1."
- "Para drift corro PSI con los thresholds 0.1 / 0.25 más una prueba KS sobre el score del
  modelo, porque los bins gruesos de PSI pueden ocultar cambios de forma dentro de un bin que en
  realidad mueven la frontera de decisión."

## 1. Métricas de clasificación: elige la métrica antes que el modelo

La métrica es una decisión de negocio, no un detalle de modelado. Enuncia primero la asimetría,
y después la fórmula.

$$\text{Precision} = \frac{TP}{TP + FP} \qquad
\text{Recall} = \frac{TP}{TP + FN} \qquad
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Precisión (precision)** domina cuando los falsos positivos son caros en operaciones, no solo
  en confianza: marcar una transacción legítima en fintech cuesta un slot de revisión humana y
  un cliente; alertar sobre un dispositivo sano en un producto de observabilidad SaaS despierta
  a un ingeniero de guardia.
- **Recall** domina cuando los falsos negativos son catastróficos: una alerta de sepsis omitida
  en salud o un patrón de lavado de dinero pasado por alto en screening KYC conlleva un costo
  regulatorio y humano que eclipsa la fatiga de alertas.
- **F1** es el default solo cuando genuinamente no puedes ordenar los dos errores; di en voz
  alta que pondera precisión y recall por igual, y que una $F_\beta$ ponderada por negocio
  (con $\beta > 1$ favoreciendo recall) es la versión honesta cuando al cliente le duelen más
  los miss.

Para problemas desbalanceados, argumenta en el plano PR, no en el plano ROC. ROC-AUC se calcula
sobre TPR y FPR; el denominador de FPR es la clase negativa enorme, así que con 0.5 por ciento
de positivos un modelo puede verse genial en ROC mientras produce una precisión inutilizable en
cada threshold plausible. El baseline de PR-AUC es la propia prevalencia, así que las ganancias
son visibles y honestas.

| Dimensión | ROC-AUC | PR-AUC |
| --- | --- | --- |
| Plano | TPR vs FPR | Precision vs Recall |
| Desbalance de clases | Excesivamente optimista: el denominador de FPR es la clase negativa enorme | Realista: la precisión se degrada a medida que los positivos se diluyen |
| Baseline del clasificador aleatorio | 0.5 sin importar la prevalencia | Igual a la prevalencia de positivos |
| Mejor para | Clases balanceadas, calidad pura de ranking | Positivos raros: fraude bajo 1 por ciento, sepsis bajo 5 por ciento |
| Punto ciego | No dice nada sobre el punto de operación | Ignora el lado de recall del ranking completo |

Dos diferenciadores senior. Primero, **MCC** (Matthews Correlation Coefficient) usa las cuatro
celdas de la matriz de confusión y se mantiene informativo bajo desbalance:

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Segundo, **la calibración**, que ninguna métrica de ranking mide. Un modelo puede tener un AUC
sobresaliente mientras sus eventos "0.8" ocurren el 55 por ciento de las veces. Mídela con el
Brier score y el Expected Calibration Error:

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2 \qquad
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\lvert \text{acc}(b) - \text{conf}(b) \rvert$$

Corrige la calibración con Platt scaling (pocos datos, sigmoide paramétrica) o regresión
isótonica (miles de muestras, no paramétrica), o temperature scaling para redes neuronales —
ajusta siempre el recalibrador sobre un holdout, nunca sobre datos de entrenamiento. Los risk
scores de salud son el caso canónico: un clínico que actúa sobre "8 por ciento de probabilidad"
consume la calibración, no el AUC.

## 2. Métricas de regresión y sus modos de fallo

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} \lvert y_i - \hat{y}_i \rvert \qquad
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

- MAE minimiza la mediana (L1); RMSE minimiza la media (L2). RMSE castiga los errores grandes
  de forma cuadrática, así que es el objetivo correcto cuando un miss grande es
  desproporcionadamente malo: una red logística que promete 3 días y tarda 9 es más del doble
  de mala que una que tarda 6.
- MAE es la métrica correcta para reportar cuando los outliers son ruido que te niegas a dejar
  que domine el KPI. Enuncia el par explícitamente: optimiza con uno, reporta con ambos, y
  agrega una métrica de quantile (Pinball Loss) cuando el producto necesita un límite superior
  y no un forecast puntual:

$$L_\tau(y, \hat{y}) = \begin{cases} \tau (y - \hat{y}) & y \ge \hat{y} \\ (1-\tau)(\hat{y} - y) & y < \hat{y} \end{cases}$$

Para forecasting de demanda, los SKU con demanda censurada (los stockouts ocultan la demanda
real) hacen que un RMSE ingenuo sea engañoso: entrena con estimaciones de demanda insesgadas o
predice el quantile del percentil 90 y mide el service level directamente.

## 3. Interioridades de gradient boosting

El modelo aditivo por etapas de Friedman: en la etapa $m$, ajusta un weak learner $h_m$ al
gradiente negativo de la pérdida evaluada en el ensemble actual, y luego da un paso reducido:

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x), \qquad
h_m \approx -\frac{\partial L(y, F(x))}{\partial F(x)}\Big|_{F = F_{m-1}}$$

donde $\nu$ es el learning rate (shrinkage). El refinamiento de XGBoost: expande la pérdida con
Taylor a segundo orden alrededor de $F_{m-1}$, de modo que cada hoja lleva una suma de
gradientes $G$ y una suma de Hessiana $H$, y el peso óptimo de hoja y la ganancia del split
quedan en forma cerrada:

$$w^* = -\frac{G}{H + \lambda}$$

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda}
- \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

con el objetivo regularizado como $\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2$
($T$ número de hojas, $\gamma$ penalización por split, $\lambda$ L2 sobre los pesos). Derivar
esto en un whiteboard separa al senior del mid-level: $\gamma$ poda los splits cuya ganancia es
negativa, $\lambda$ encoge los pesos de las hojas, y el subsampling de filas/columnas suma
regularización estocástica por encima.

| Dimensión | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| Búsqueda de splits | Pre-sorted / quantile sketch | Basado en histogramas, poco uso de memoria | Árboles oblivious (simétricos) |
| Estrategia de crecimiento | Level-wise | Leaf-wise (guiado por la pérdida) | Level-wise, profundidad simétrica |
| Features categóricas | One-hot / target encoding aguas arriba | Nativas vía estadísticas de gradiente | Nativas, ordered target statistics (resistentes a leakage) |
| Perfil de velocidad | Baseline | El más rápido a millones de filas | Más lento; el más fuerte en datos ricos en categóricas |
| Riesgo de overfitting | Moderado, requiere tuning | Mayor con crecimiento leaf-wise; limita `num_leaves` | Menor: los árboles simétricos regularizan estructuralmente |
| Cuándo elegirlo | Tabular general, el ecosistema más amplio | Tabular a gran escala, reentrenamiento rápido | Muchas categóricas de alta cardinalidad, preocupaciones estrictas de leakage |

Los missing values se manejan nativamente en XGBoost aprendiendo una dirección de split default
por nodo —di esto con exactitud, porque "imputa automáticamente" es incorrecto. Las monotonic
constraints están disponibles en los tres y son un diferenciador genuino en scoring crediticio
y clínico, donde un regulador preguntará por qué el score cayó cuando el ingreso subió.

## 4. Detección de anomalías

**Isolation Forest** construye árboles aleatorios paralelos a los ejes. La longitud de camino
esperada para aislar un punto, $E[h(x)]$, es corta para las anomalías porque viven en regiones
dispersas; el score la normaliza contra la longitud de camino promedio $c(n)$ de una búsqueda
fallida en un árbol de búsqueda binario de $n$ muestras ($H(i) \approx \ln(i) + 0.5772$, la
constante de Euler-Mascheroni):

$$s(x, n) = 2^{-E[h(x)] / c(n)}, \qquad c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

$s \to 1$ cuando $E[h(x)] \to 0$ (anómalo), $s \to 0.5$ para los inliers. Sin métrica de
distancia, sin estimación de densidad, sublineal al momento de puntuar —por eso es el tripwire
por defecto para streams de transacciones y telemetría de dispositivos.

**Los autoencoders** aprenden una reconstrucción de baja dimensión del comportamiento normal;
el error de reconstrucción $\lVert x - \hat{x} \rVert^2$ marca los puntos que la manifold no
puede representar. Brillan con señales estructuradas (ventanas de sensores, imágenes) pero
fallan en silencio: si las anomalías son más que una fracción pequeña de los datos de
entrenamiento, el modelo aprende a reconstruirlas también, y tu alertamiento muere sin error.

| Dimensión | Isolation Forest | Autoencoder | Distribucional (PSI/KS sobre scores) |
| --- | --- | --- | --- |
| Supuesto | Las anomalías se aíslan en menos splits aleatorios | Los datos normales viven en una manifold aprendible | La distribución de salida es estacionaria |
| Requerimiento de datos | Sin etiquetas, funciona con tamaños de muestra modestos | Corpus grande de muestras normales | Ventana de referencia de scores |
| Interpretabilidad | Longitud de camino por punto | Error de reconstrucción por feature | Distancia al threshold, fácil de graficar |
| Costo en streaming | $O(\log n)$ por punto | Un forward pass | Binning $O(1)$ |
| Fallo característico | Anomalías locales dentro de clusters densos puntúan bajo | Aprende a reconstruir anomalías si se contamina | Bins gruesos ocultan drift dentro del bin |
| Cuándo elegirlo | Primer baseline en streams tabulares | Señales estructuradas ricas con estructura explotable | Guardrail por capas sobre cualquier detector |

## 5. Sistemas de recomendación y embeddings

La factorización de matrices clásica aprende factores latentes de usuarios e ítems en conjunto;
alternating least squares (ALS) resuelve cada lado en forma cerrada mientras mantiene fijo el
otro:

$$\min_{X, Y} \sum_{(u,i) \in K} \left( r_{ui} - x_u^\top y_i \right)^2
+ \lambda \left( \lVert x_u \rVert^2 + \lVert y_i \rVert^2 \right)$$

El collaborative filtering explota el comportamiento pero colapsa en cold start; el filtrado
basado en contenido explota los metadatos de los ítems y maneja ítems nuevos, pero limita el
descubrimiento (solo llega a recomendar más de lo que al usuario ya le gustó). Los recommenders
de producción son de dos etapas: un generador de candidatos rápido (ANN sobre embeddings, o
ALS) propone cientos de ítems, y luego un ranker más pesado (gradient boosting sobre features,
o un cross-encoder) los ordena —el mismo split recall/rerank que el RAG moderno, y vale la pena
nombrarlo en la entrevista.

La similitud coseno es la primitiva compartida entre recommenders y búsqueda semántica:

$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert}$$

Vigila el sesgo de popularidad offline (un NDCG@k ingenuo puede maximizarse recomendando
blockbusters) y trata las métricas offline como un filtro, no como un oráculo: recall@k y
NDCG@k para hacer gate de candidatos, y luego un A/B online con métricas guardrail (diversidad
de sesión, retención a largo plazo) para confirmar.

## 6. Drift: PSI y KS como guardrail permanente

El data drift, no la decadencia del modelo, es la causa dominante de degradación silenciosa en
los modelos clásicos desplegados. El Population Stability Index sobre $B$ bins, con $e_i$ la
proporción de referencia y $a_i$ la proporción en vivo del bin $i$:

$$\text{PSI} = \sum_{i=1}^{B} (a_i - e_i) \ln \frac{a_i}{e_i}$$

Thresholds operativos estándar, y el contrato exacto implementado en
`src/interview_prep/drift.py`: por debajo de 0.1 estable, de 0.1 a 0.25 moderado, por encima de
0.25 severo. El estadístico Kolmogorov-Smirnov de dos muestras es el chequeo complementario:

$$D_{n,m} = \sup_x \lvert F_{1,n}(x) - F_{2,m}(x) \rvert$$

PSI depende de la resolución de bins y puede ocultar un shift concentrado dentro de un bin
grueso; KS es sensible al mayor hueco de la CDF en cualquier punto del soporte. Corre ambos,
sobre los inputs que importan y sobre el score del modelo mismo, y empareja las alertas de
drift con monitoreo de calibración —una población shiftada suele recalibrar la distribución del
score antes de mover el AUC.

## 7. Explicaciones: SHAP values

SHAP asigna a cada feature $i$ una contribución aditiva $\phi_i$ promediando su contribución
marginal sobre todas las coaliciones de features $S \subseteq F \setminus \{i\}$:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}}
\frac{\lvert S \rvert! \, (\lvert F \rvert - \lvert S \rvert - 1)!}{\lvert F \rvert!}
\left( v(S \cup \{i\}) - v(S) \right)$$

Las propiedades que le importan a un regulador son efficiency (las contribuciones más el valor
base suman exactamente la predicción) y consistency (si el efecto marginal de una feature
crece, su atribución no puede encogerse). En la práctica: TreeSHAP para valores exactos en
tiempo polinómico sobre ensembles de árboles; sé honesto en que las features correlacionadas
reparten el crédito de forma arbitraria, y empareja SHAP local con resúmenes globales y
ejemplos contrafactuales.

## Referencia cruzada con el repositorio

Los conceptos anteriores están implementados en `src/interview_prep/traditional_ml.py` y
`src/interview_prep/drift.py`:

```python
from interview_prep.traditional_ml import MockAnomalyDetector, MockRecommenderSystem

detector = MockAnomalyDetector(threshold=2.5)
score = detector.compute_anomaly_score([38.1, 36.9, 37.2, 41.5])  # z-score proxy
flagged = detector.is_anomaly([38.1, 36.9, 37.2, 44.0])           # score > threshold

recommender = MockRecommenderSystem(
    item_embeddings={"care-plan-A": [0.9, 0.1], "care-plan-B": [0.2, 0.8]}
)
top = recommender.recommend([0.8, 0.2], top_k=2)  # cosine-similarity ranking
```

```python
from interview_prep.drift import drift_verdict, ks_statistic, population_stability_index

psi = population_stability_index(expected_scores, live_scores)
if drift_verdict(psi) != "STABLE":          # 0.1 / 0.25 thresholds
    ks = ks_statistic(expected_scores, live_scores)  # localize the shift
```

## Historias de guerra en producción / Errores frecuentes

- **Threshold heredado del análisis ROC.** Un modelo de fraude se desplegó con el cutoff
  default de 0.5; la cola de revisión recibió 40 veces la capacidad diaria del equipo. Causa
  raíz: la métrica (AUC) se eligió sin análisis del punto de operación. Mitigación: elige el
  threshold desde la curva PR bajo una restricción de capacidad, monitorea la precisión en ese
  threshold y reajústalo en cada reentrenamiento.
- **Leakage por joins con event-time.** Un modelo de readmisión puntuó 0.94 AUC offline y 0.61
  online. Causa raíz: una feature (dispensaciones de medicamentos posteriores al alta) solo se
  conocía después del momento de la predicción. Mitigación: point-in-time joins con semántica
  de event-time de un feature store, y un leak scan que recalcula cada feature a la fecha del
  timestamp de la etiqueta.
- **Splits aleatorios en series de tiempo.** Un modelo de demanda entrenado con 12 meses que
  incluían diciembre y probado en enero lucía brillante y falló en el lanzamiento. Causa raíz:
  la estacionalidad más la no estacionariedad rompieron el supuesto i.i.d. Mitigación:
  evaluación rolling-origin (walk-forward) y reporte de MAE/RMSE por horizonte.
- **SMOTE aplicado antes del split.** Los vecinos sintéticos de pacientes del test set se
  filtraron al entrenamiento, inflando el recall en 15 puntos. Causa raíz: resampling fuera del
  fold de entrenamiento. Mitigación: remuestrea dentro de un pipeline ajustado por fold, o usa
  class weights, que no pueden tener leakage.
- **PSI plano mientras el modelo decaía.** Cinco bins de quantiles gruesos ocultaron un shift
  concentrado en el decil superior de la distribución del score. Causa raíz: resolución de
  bins. Mitigación: 10-20 bins sobre el score, KS como segunda opinión, y PSI a nivel de
  segmento para los segmentos en los que el negocio realmente actúa.
- **Drift de threshold tras un swap champion-challenger.** El nuevo champion estaba calibrado
  de forma distinta; el cutoff intacto de 0.35 colapsó el recall. Causa raíz: thresholds
  tratados como constantes y no como artefactos del modelo. Mitigación: guarda el threshold
  junto con el artefacto del modelo y haz gate de los releases con métricas de calibración en
  CI.

## Lo que los entrevistadores realmente preguntan

- **"Tu modelo de fraude tiene 0.97 de ROC-AUC pero el cliente lo llama inútil. Diagnostica."**
  - Pide prevalencia y punto de operación antes de tocar el modelo.
  - Causas probables: mala calibración, threshold fijado arbitrariamente, precisión inutilizable
    a la capacidad de revisión, o leakage de evaluación inflando el AUC.
  - Deliverable: curva PR, precisión en el threshold factible de capacidad, gráfico de
    calibración.
- **"¿Cómo elige un split XGBoost?"**
  - Expansión de Taylor de segundo orden de la pérdida; peso de hoja en forma cerrada
    $-G/(H+\lambda)$.
  - La fórmula de Gain compara izquierda, derecha y padre; quantile sketches aproximados para
    escalar; dirección default consciente de sparsity para missing values.
- **"¿Cuándo no usarías gradient boosting?"**
  - Texto crudo, imágenes, audio (se necesita representation learning); modelos en streaming
    que requieren actualizaciones incrementales baratas; latencia ultrabaja con miles de
    features (considera modelos lineales destilados); necesidades estrictas de extrapolación
    (los árboles no extrapolan).
  - Contraargumento para ofrecer voluntariamente: las monotonic constraints y la
    interpretabilidad vía TreeSHAP hacen a los GBM aceptables incluso en scoring regulado.
- **"¿Isolation Forest o autoencoder para anomalías de telemetría?"**
  - Supuesto de pocas-y-diferentes vs supuesto de manifold; eficiencia de muestra; costo en
    streaming.
  - Respuesta práctica: Isolation Forest como tripwire, autoencoder para atribución por sensor,
    chequeos distribucionales (PSI/KS) como meta-guardrail.
- **"¿Cómo evalúas un recommender offline?"**
  - Métricas de dos etapas: recall@k para candidatos, NDCG@k para ranking; splits
    leave-last-out, nunca splits aleatorios, en datos temporales.
  - Protégete del sesgo de popularidad; lo offline es un filtro, el A/B online decide.
- **"El modelo está mal calibrado. Arréglalo."**
  - Mide ECE/Brier por segmento; Platt para pocos datos, isótonica para más, temperature
    scaling para redes neuronales; reajusta sobre holdout; re-verifica después de cada
    reentrenamiento.
- **"Explica una predicción individual a un regulador."**
  - SHAP con un baseline documentado; la propiedad de efficiency garantiza que la
    descomposición suma; enuncia el caveat de features correlacionadas y emparéjalo con
    contrafactuales.

## Lecturas recomendadas

- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (Annals of
  Statistics, 2001) — https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full
- Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System" (KDD 2016) —
  https://arxiv.org/abs/1603.02754
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017)
  — https://arxiv.org/abs/1711.08789
- Prokhorenkova et al., "CatBoost: Unbiased Boosting with Categorical Features" (NeurIPS
  2018) — https://arxiv.org/abs/1706.09516
- Liu, Ting, and Zhou, "Isolation Forest" (ICDM 2008) —
  https://www.cs.uic.edu/~zhangxz/publ/papers/isof.pdf
- Saito and Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot
  When Evaluating Binary Classifiers on Imbalanced Datasets" (PLOS ONE, 2015) —
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432
- Guo, Pleiss, Sun, and Weinberger, "On Calibration of Modern Neural Networks" (ICML 2017)
  — https://arxiv.org/abs/1706.04599
- Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
  — https://arxiv.org/abs/1705.07874
- Koren, Bell, and Volinsky, "Matrix Factorization Techniques for Recommender Systems"
  (IEEE Computer, 2009) — https://ieeexplore.ieee.org/document/5197422
