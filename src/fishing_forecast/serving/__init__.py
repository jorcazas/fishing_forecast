"""Serving: API mínima de inferencia (CQR) sobre las zonas del pool.

`forecast.build_store()` entrena la CQR de producción (misma lógica que Exp 4) una vez y
cachea, por serie (especie × unidad económica), el pronóstico calibrado (mediana + bandas
80/90%) para el periodo de prueba. `api` lo expone vía FastAPI + un front mínimo.
"""
