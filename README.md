# Prueba-tecnica-Anfler

## Objetivo
Este repositorio resuelve los dos ejercicios de la prueba tecnica:
1. Experimento de modelos resumidores con evaluacion automatica.
2. Analisis del clasificador de ruteo semantico.

## Requisitos
- Python 3.12+
- Dependencias de requirements.txt

## Instalacion
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Ejercicio 1 - Runner del experimento
Archivo principal: `runner.py`

### Modo auto (recomendado)
Usa API real si detecta `OPENAI_API_KEY`; si no existe, cae automáticamente a simulacion.

```bash
.venv\\Scripts\\python.exe runner.py --mode auto
```

### Modo simulacion (sin API key)
Permite validar flujo completo, archivos de salida y seleccion de ganador sin consumir API.

```bash
.venv\\Scripts\\python.exe runner.py --simulate
```

### Control de alcance (escalable a 1000 documentos)
Por defecto ejecuta 1 fila (requisito de la prueba), pero el runner ya soporta múltiples filas o todas:

```bash
# 5 filas
.venv\\Scripts\\python.exe runner.py --rows 5 --mode simulate

# todas las filas del dataset (aplicable al dataset completo de 1000)
.venv\\Scripts\\python.exe runner.py --all-rows --mode real
```

Salidas generadas en `outputs/`:
- `summary_experiment_results.json`
- `summary_experiment_results.csv`
- `summary_experiment_report.md`

### Modo real (con API key)
1. Crear `.env` con:

```env
OPENAI_API_KEY=tu_api_key
```

2. Ejecutar:

```bash
.venv\\Scripts\\python.exe runner.py
```

Tambien se puede forzar modo real con:

```bash
.venv\\Scripts\\python.exe runner.py --mode real
```

## Ejercicio 1 - Notebook de experimentos
Notebook: `experimentos.ipynb`

Incluye ejecucion del experimento en modo simulacion, lectura de resultados y pruebas de robustez/escalabilidad para respaldar decisiones.
Tambien incluye un smoke test real de API key que toma una fila del dataset tecnico y evalua contra `puntos_criticos`, guardando evidencia en `outputs/api_key_smoke_test.json`.

## Ejercicio 2 - Analisis del clasificador
Notebook: `analisis_clasificador.ipynb`

Incluye:
- Accuracy global
- Reporte por clase (precision/recall/F1)
- Matriz de confusion
- Analisis de errores
- Lectura de contexto multilabel para distinguir errores estrictos vs rutas plausibles

## Validacion automatica sin API key
Se incluyen pruebas offline para el Ejercicio 1:

```bash
.venv\\Scripts\\python.exe -m unittest tests\\test_runner_simulation.py -v
```

## Validacion integral en un comando
Para validar ambos ejercicios (E1 simulado + metricas E2):

```bash
.venv\\Scripts\\python.exe validate_all.py
```

Genera `outputs/validation_report.json` con el resultado consolidado.

Incluye un bloque `approval` con resultado `pass/fail` usando umbrales por defecto:
- `min_accuracy = 0.80`
- `min_macro_f1 = 0.75`
- `min_winner_quality = 70.0`

Si falla algún criterio, el script termina con código de salida `1`.

Tambien se pueden ajustar umbrales:

```bash
.venv\\Scripts\\python.exe validate_all.py --min-accuracy 0.82 --min-macro-f1 0.76 --min-winner-quality 75
```
