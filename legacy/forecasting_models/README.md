# Repositorio del Paper: Predicción del volumen de pesca de langosta en San Quintin, Baja California

Este repositorio contiene el código y los scripts asociados con el paper "Predicción del volumen de pesca de langosta en San Quintin, Baja California". El objetivo principal del paper es explorar y desarrollar modelos de pronóstico aplicados a la pesca en comunidades de San Quintín, Baja California.

## Contenido del Repositorio

- Scripts de Modelado: La carpeta models contiene los scripts utilizados para desarrollar y entrenar modelos de pronóstico. Estos scripts abarcan desde la preparación de datos hasta la evaluación de modelos.

- ETL: La carpeta etl almacena los scripts de ETL para la extracción, limpieza y carga de los datos utilizados para este proyecto.

- Datos: La carpeta data almacena los conjuntos de datos utilizados en el estudio. Incluye datos oceanográficos, registros de arribos pesqueros y otras variables ambientales. Los datos almacenados aquí ya fueron procesados por los procesos de ETL.

- Resultados: La carpeta results almacena los resultados obtenidos durante la evaluación de modelos, como métricas de rendimiento y visualizaciones.

## Instrucciones de Uso
1. Clonar el Repositorio:

~~~bash
git clone https://github.com/tu_usuario/repo-pronostico-pesca-baja-california.git
~~~
2. Configuración del Entorno:
Se recomienda crear un entorno virtual e instalar las dependencias necesarias.

~~~bash
cd repo-pronostico-pesca-baja-california
python -m venv venv
source venv/bin/activate  # En sistemas basados en Unix
# o
.\venv\Scripts\activate  # En sistemas Windows
pip install -r requirements.txt
~~~
3. Ejecutar Scripts:
Explore la carpeta scripts y ejecute los scripts en el orden adecuado para replicar el entrenamiento y evaluación de modelos.

4. Explorar Resultados:
Revise la carpeta results para acceder a métricas de rendimiento, visualizaciones y cualquier resultado relevante.

## Detalles del Paper
- Título: Predicción del volumen de pesca de langosta en San Quintin, Baja California
- Autor: José Javier Orcazas Leal
- Fecha de Publicación: 04 de Diciembre de 2024
- Este repositorio sirve como recurso para acceder al código fuente y datos utilizados en el desarrollo del paper. Se alienta a los interesados a explorar, replicar y construir sobre este trabajo.

Palabras Clave: Pronóstico, Pesca, Baja California, Aprendizaje de Máquina, Modelos Estadísticos, Oceanografía.