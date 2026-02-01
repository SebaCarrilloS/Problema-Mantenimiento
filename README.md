# Pipeline de Mantenimiento Predictivo (End-to-End)

Este proyecto implementa un **pipeline completo de mantenimiento predictivo**, cubriendo todo el ciclo de vida: desde la preparación de datos y el modelamiento con machine learning, hasta la **optimización de planes de mantenimiento** y el **monitoreo en producción (MLOps)**.

El foco no es solo predecir fallas, sino **apoyar la toma de decisiones operacionales** bajo restricciones reales de capacidad y presupuesto.

---

## 1. Contexto del problema

En entornos industriales, las fallas de equipos generan altos costos operacionales y pérdidas de disponibilidad. Sin embargo:

- No es posible realizar mantenimiento preventivo a todos los equipos.
- Los recursos de mantenimiento son limitados.
- Las predicciones deben traducirse en **acciones concretas**, no solo en scores.

Este proyecto aborda el mantenimiento predictivo como un **problema de decisión**, no únicamente como un problema de clasificación.

---

## 2. Dataset

Se utiliza el **dataset de Mantenimiento Predictivo de Microsoft Azure**, que contiene:

- Telemetría de alta frecuencia
- Registros de errores
- Historial de mantenciones
- Eventos de falla
- Metadatos de las máquinas

Los datos son agregados a nivel **diario por máquina**, generando un dataset temporal apto para modelamiento, optimización y monitoreo.

---

## 3. Enfoque técnico

El pipeline se estructura en etapas desacopladas pero conectadas:

1. **Exploración y calidad de datos**
2. **Ingeniería de variables**
3. **Modelamiento de riesgo de falla**
4. **Optimización del plan de mantenimiento**
5. **Monitoreo y detección de drift**

Cada etapa genera artefactos reutilizables por las siguientes fases.

---

## 4. Arquitectura del pipeline

a. Datos Crudos
b. Ingeniería de Variables
c. Modelo de Riesgo de Falla (ML)
d. Scoring Operativo
e. Optimización de Mantenimiento (Costo-Aware)
f. Monitoreo y Detección de Drift


El pipeline puede ejecutarse mediante **notebooks** (análisis y explicación) o mediante una **interfaz de línea de comandos** (`main.py`) para ejecución reproducible.

---

## 5. Modelamiento

- **Modelo base**: Regresión Logística
- **Énfasis**: calibración de probabilidades y estabilidad temporal
- **Estrategia de validación**: split temporal (sin leakage)
- **Métricas**:
  - ROC-AUC
  - Average Precision (AP)

El modelo entrega **probabilidades de falla**, no decisiones binarias.

---

## 6. Optimización del mantenimiento

Las probabilidades de falla se transforman en decisiones operativas mediante un enfoque de **optimización bajo restricciones**.

Restricciones consideradas:
- Capacidad máxima de mantención
- Presupuesto disponible
- Trade-off entre costo de mantención preventiva y costo de falla

El resultado es un **plan de mantenimiento óptimo** que minimiza el costo esperado total.

---

## 7. Monitoreo y MLOps

El proyecto incorpora una capa de monitoreo que evalúa:

- **Drift de variables** (PSI, KS)
- **Drift de scores**
- **Drift de etiquetas**
- **Degradación de desempeño**

Se utiliza un **sistema de semáforo** (VERDE / AMARILLO / ROJO) que resume el estado del modelo y recomienda acciones como monitoreo reforzado o reentrenamiento.

---

## 8. Ejecución del proyecto


### Configuración del entorno

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt


Comandos disponibles

python main.py --help

### Configuración del entorno

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py --help

python main.py train     # Entrena el modelo y guarda artefactos
python main.py score     # Genera scoring operativo
python main.py optimize  # Optimiza el plan de mantenimiento
python main.py monitor   # Ejecuta monitoreo y detección de drift

---

## 9. Resultados y artefactos generados

La ejecución del pipeline produce artefactos reproducibles que permiten auditar, analizar y operar el sistema de mantenimiento predictivo:

- **Modelos entrenados**
  - Modelos de machine learning serializados (`.joblib`)
  - Modelos calibrados para uso operativo

- **Scoring operativo**
  - Probabilidades de falla por máquina y período
  - Datos listos para priorización de mantenimiento

- **Optimización de mantenimiento**
  - Plan óptimo de mantenimiento (`.csv`)
  - Resumen de costos, beneficios y reducción de riesgo (`.json`)
  - Comparación entre escenario con y sin intervención

- **Monitoreo y MLOps**
  - Reportes de drift de variables (PSI, KS)
  - Reportes de drift de scores y etiquetas
  - Métricas de desempeño en el tiempo
  - Resumen de monitoreo con sistema de semáforo (VERDE / AMARILLO / ROJO)

Estos artefactos permiten separar claramente las etapas de **modelamiento**, **decisión** y **monitoreo**, facilitando su uso en contextos productivos.

---

## 10. Limitaciones y próximos pasos

### Limitaciones actuales
- Horizonte de predicción fijo para la ocurrencia de fallas
- Supuestos de costos estáticos para mantención y fallas
- Monitoreo ejecutado en modalidad batch
- Optimización simplificada respecto a calendarios y recursos reales

### Próximos pasos y extensiones posibles
- Incorporar modelos de **supervivencia** o **Remaining Useful Life (RUL)**
- Modelar costos dinámicos y escenarios de incertidumbre
- Integrar planificación de mantenimiento a nivel de calendario
- Implementar monitoreo continuo o en streaming
- Automatizar reentrenamiento ante alertas persistentes de drift

Estas extensiones permitirían acercar aún más el sistema a un **despliegue industrial de producción a gran escala**.


## 11. Conclusión

Este proyecto muestra cómo el mantenimiento predictivo puede abordarse como un **sistema end-to-end orientado a la toma de decisiones**, y no únicamente como un ejercicio de modelamiento predictivo.

La integración de **machine learning**, **optimización bajo restricciones** y **monitoreo MLOps** permite cerrar el ciclo completo entre datos, predicciones y acción operativa, acercando el enfoque a un entorno industrial real.

El pipeline resultante es **reproducible, auditable y extensible**, y sienta las bases para futuras evoluciones hacia soluciones de mantenimiento predictivo a escala productiva.
