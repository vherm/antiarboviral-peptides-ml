# Anti-arboviral peptide activity prediction using machine learning

Repositorio de código de investigación asociado a una tesis de maestría sobre predicción de actividad antiarboviral de péptidos mediante aprendizaje automático.

**Estado: borrador de publicación; requiere revisión y pruebas antes de hacerlo público.**

## Contenido

- `01_dataset_construction/`: seis métodos definitivos de selección de negativos y utilidades auxiliares originales.
- `02_feature_extraction/`: reservado para los scripts originales de extracción iLearnPlus y ESM-2 (pendientes).
- `03_model_pipeline/`: preparación, entrenamiento, evaluación, reportes e interpretabilidad.
- `04_experiment_orchestration/`: orquestación de ejecuciones.
- `config/`: ejemplos de configuración y archivos de entorno aportados.
- `docs/REPRODUCIBILITY.md`: alcance y advertencias metodológicas.

## Métodos de selección de negativos

1. Aleatorio.
2. Distancia de Levenshtein.
3. Distancia euclidiana mínima con iLearnPlus.
4. Distancia coseno mínima con ESM-2.
5. Agrupamiento basado en positivos y distancia euclidiana con iLearnPlus.
6. Agrupamiento basado en positivos y distancia coseno con ESM-2.

## Requisitos y ejecución

Consultar `config/environment.yml`, `config/requirements.txt` y las cabeceras/argumentos de cada script. Los archivos provienen de componentes diferentes y **todavía no se ha verificado un entorno único compatible ni una ejecución integral**. No se han modificado las implementaciones originales.

## Datos

Los datasets, embeddings y resultados voluminosos no se incluyen en este paquete inicial. Debe verificarse su licencia y procedencia antes de distribuirlos.

## Citación y licencia

`CITATION.cff` contiene datos mínimos de citación pendientes de completar con la referencia bibliográfica de la tesis. La licencia de distribución queda pendiente de elección.
