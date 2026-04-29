"""fishing_forecast — predicción del volumen de pesca de langosta en San Quintín, BC.

Paquete principal de la fase de expansión 2026 del proyecto. Vive la implementación
nueva de ETL, features, modelos y evaluación. El código del borrador 2024 (`etl/`,
`forecasting_models/` en la raíz) se mantiene como legacy hasta validar este pipeline.
"""

# Hacer que urllib3/requests usen el trust store del SO. Necesario porque algunos
# servidores .gob.mx mandan cadenas SSL incompletas (sin intermedio); el SO sabe
# cómo resolver eso vía AIA, certifi por sí solo no.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # truststore es dep core; este except solo cubre instalaciones rotas
    pass

__version__ = "0.1.0"
