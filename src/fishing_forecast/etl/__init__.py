"""ETL: extracción, transformación, agregación y consolidación de datos.

Etapas (ver docs/etl_design.md §3):

- `extract/`     : raw → disco (data/raw/...)
- `transform/`   : raw → interim long-tidy (data/interim/<fuente>/...)
- `aggregate/`   : interim → interim por UE (ocean_by_ue, mhw)
- `consolidate`  : interim → data/processed/dataset_vN.parquet
- `quality_checks` : asserts y schemas
"""
