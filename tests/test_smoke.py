"""Smoke tests del bootstrap. Verifican que el paquete carga y la config funciona."""

from __future__ import annotations

import fishing_forecast
from fishing_forecast.config import get_settings
from fishing_forecast.utils.dates import in_season, season_id


def test_package_version():
    assert fishing_forecast.__version__ == "0.1.0"


def test_settings_loads():
    settings = get_settings()
    assert settings.data_root.name == "data"
    assert settings.raw_dir.name == "raw"


def test_season_id_in_season():
    from datetime import date

    assert season_id(date(2018, 11, 1)) == "2018_2019"
    assert season_id(date(2019, 1, 15)) == "2018_2019"
    assert season_id(date(2019, 2, 15)) == "2018_2019"  # último día de temporada
    assert season_id(date(2018, 9, 15)) == "2018_2019"  # primer día


def test_season_id_between_seasons():
    from datetime import date

    # Días en el gap deben asignarse a la próxima temporada (útil para 90d shift).
    assert season_id(date(2019, 6, 1)) == "2019_2020"
    assert season_id(date(2019, 9, 14)) == "2019_2020"
    assert season_id(date(2019, 2, 16)) == "2019_2020"  # día siguiente al fin de temporada


def test_in_season_lobster_calendar():
    from datetime import date

    # langosta-SQ: 09-15 → 02-15 (cruza año)
    assert in_season(date(2019, 12, 1), 9, 15, 2, 15)
    assert in_season(date(2019, 9, 15), 9, 15, 2, 15)
    assert in_season(date(2020, 2, 15), 9, 15, 2, 15)
    assert not in_season(date(2019, 6, 1), 9, 15, 2, 15)
    assert not in_season(date(2019, 9, 14), 9, 15, 2, 15)
