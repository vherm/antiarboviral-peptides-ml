# Reproducibilidad y alcance

Este paquete reúne los scripts Python identificados en los archivos fuente de construcción de datasets, pipeline y orquestador. **No es todavía una distribución validada de extremo a extremo.**

- Conjunto definitivo auditado: 247 secuencias positivas y 1.235 negativas por dataset experimental.
- Las repeticiones de ciertos métodos comparten secuencias negativas; no deben tratarse automáticamente como observaciones independientes.
- Los métodos euclidianos de construcción utilizan características iLearnPlus y los de coseno representaciones ESM-2; esta diferencia debe considerarse en comparaciones.
- Comprobar las rutas de archivos y dependencias antes de ejecutar.
- Revisar que los datos puedan redistribuirse antes de publicarlos.
- Comprobar solapamiento entre entrenamiento y prueba, duplicados y ajuste de preprocesamiento dentro de cada partición.
- Fijar versión del código, entorno y semilla para cada resultado reportado.
