# MobileAIBench - Evaluador de Modelos LLM en iOS

Aplicación iOS para evaluar el rendimiento y precisión de Large Language Models (LLMs) ejecutándose localmente en dispositivos móviles Apple.

## Descripcion

**MobileAIBench** es una herramienta de benchmarking diseñada para medir el rendimiento de modelos de lenguaje cuantizados en formato GGUF ejecutándose directamente en dispositivos iOS. La aplicación utiliza el framework llama.cpp para realizar inferencia on-device sin necesidad de conexión a servidores externos.

### Funcionalidades principales

- Carga y evaluación de modelos GGUF cuantizados directamente en iOS
- Benchmarking contra múltiples datasets de NLP
- Métricas de rendimiento: tokens por segundo, tiempo de carga, uso de memoria
- Métricas de precisión por tarea: Exact Match (EM), F1 Score, ROUGE, SQL parsing
- Persistencia y exportación de resultados en JSON


## Modelos Evaluados

Los modelos utilizados en este proyecto están disponibles en **[Hugging Face](https://huggingface.co/chaparro2001/models)**.

### Modelos principales evaluados

| Modelo | Cuantizaciones |
|--------|---------------|
| **Gemma 3 4B Instruct** | Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S |
| **Qwen 3 4B Instruct** | Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S |

### Modelos adicionales

- Phi-2 (Q4_K_M)
- Gemma 2B Instruct (Q4_K_M)


## Datasets de Evaluación

### Tareas textuales

| Dataset | Tarea | Métricas |
|---------|-------|----------|
| **HotpotQA** | Question Answering con contexto | Exact Match, F1 Score |
| **Databricks Dolly** | Seguimiento de instrucciones | Exact Match, F1 Score |
| **SQL Create Context** | Generación de consultas SQL | SQL Parser F1, Distancia Levenshtein |
| **Edinburgh XSum** | Resumen abstractivo | ROUGE-1, ROUGE-L |


## Métricas de Rendimiento

La aplicación captura las siguientes métricas para cada evaluación:

- **Model Load Time**: Tiempo de carga del modelo en memoria (ms)
- **Prompt Tokens/sec**: Velocidad de procesamiento del prompt de entrada
- **Time to First Token (TTFT)**: Latencia hasta generar el primer token
- **Output Tokens/sec**: Velocidad de generación de tokens de salida
- **Average Eval Time**: Tiempo medio de evaluación por ejemplo


## Resultados

Los resultados de los benchmarks se almacenan en formato JSON en la carpeta `Resultados/`.

### Ejemplo de resultado (Gemma 3 4B Q3_K_M en HotpotQA)

```json
{
  "modelName": "gemma-3-4b-it-q3_k_m",
  "numberOfExamples": 10,
  "benchmarkMetrics": {
    "model_load_time": 8.38,
    "averagePromptTokens": 635,
    "averagePromptTokenPerSec": 160.4,
    "averageEvalTokenPerSec": 11.5,
    "averageTotalTime": 5.33
  }
}
```

## Tecnologías

- **SwiftUI** - Framework de interfaz de usuario
- **llama.cpp** - Motor de inferencia LLM en C++

## Plataformas Soportadas

- iOS (arm64)

## Uso

1. Descargar los modelos GGUF desde [Hugging Face](https://huggingface.co/chaparro2001/models)
2. Copiar los modelos a la carpeta Documents de la aplicación en el dispositivo
3. Seleccionar modelo, dataset y número de ejemplos en la interfaz
4. Ejecutar el benchmark
5. Los resultados se guardan automáticamente en formato JSON

## Autor

Juan Chaparro García - Trabajo Fin de Máster

## Referencias

Este proyecto se basa en y/o utiliza código de los siguientes repositorios:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** - Motor de inferencia LLM en C/C++ utilizado para ejecutar modelos en dispositivos móviles
- **[MobileAIBench iOS App](https://github.com/SalesforceAIResearch/MobileAIBench/tree/main/ios-app)** - Aplicación de referencia para benchmarking de modelos de IA en dispositivos móviles

## Licencia

Este proyecto forma parte de un Trabajo Fin de Máster (TFM) con fines académicos y de investigación.
