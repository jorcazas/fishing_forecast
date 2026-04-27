"""Helpers de fechas: temporadas pesqueras y banderas in_season.

Una temporada se identifica por su año de inicio. Para langosta-SQ con calendario
09-15 → 02-15, la temporada `2017_2018` cubre 2017-09-15 → 2018-02-15.

Cuando una fecha cae fuera del rango de temporada (días "gap"), se asigna a la
**próxima** temporada — útil porque las features oceanográficas previas a una
temporada (típicamente con desplazamiento de 90 días) sirven para predecirla.
"""

from __future__ import annotations

from datetime import date


def in_season(
    d: date,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> bool:
    """True si `d` cae dentro del rango (inclusive en ambos extremos).

    El rango puede cruzar año (start > end implica wrap-around). Para langosta-SQ:
    `in_season(d, 9, 15, 2, 15)`.
    """
    md = (d.month, d.day)
    start = (start_month, start_day)
    end = (end_month, end_day)
    if start <= end:
        return start <= md <= end
    return md >= start or md <= end


def season_id(
    d: date,
    start_month: int = 9,
    start_day: int = 15,
    end_month: int = 2,
    end_day: int = 15,
) -> str:
    """Identificador de la temporada asociada a `d`.

    - Si `d` cae dentro del rango de temporada → identificador de esa temporada.
    - Si `d` cae en el "gap" entre temporadas → identificador de la *próxima* temporada.

    Para temporadas que cruzan año (langosta-SQ 09-15 → 02-15) el id es `YYYY_YYYY+1`
    donde el primer YYYY es el año de inicio (septiembre).
    Para temporadas dentro de un año (e.g. 01-01 → 05-31) el id colapsa a `YYYY_YYYY`.

    Ejemplos langosta-SQ (start=9-15, end=2-15):
      date(2017, 11, 1)  → "2017_2018"  (en temporada)
      date(2018, 1, 31)  → "2017_2018"  (en temporada, cross-year)
      date(2018, 2, 15)  → "2017_2018"  (último día de temporada)
      date(2018, 6, 1)   → "2018_2019"  (gap; asignar próxima)
      date(2018, 9, 15)  → "2018_2019"  (primer día de temporada)
    """
    crosses_year = (start_month, start_day) > (end_month, end_day)
    start_this = date(d.year, start_month, start_day)
    end_this = date(d.year, end_month, end_day)

    if crosses_year:
        # Temporada cruza año: arranca en (start_month, start_day) y termina en
        # (end_month, end_day) del año siguiente.
        if d >= start_this:
            return f"{d.year}_{d.year + 1}"
        if d <= end_this:
            # Carry-over de la temporada anterior.
            return f"{d.year - 1}_{d.year}"
        # Gap entre temporadas (después del end de este año, antes del start).
        return f"{d.year}_{d.year + 1}"

    # Temporada dentro de un año calendario.
    if start_this <= d <= end_this:
        return f"{d.year}_{d.year}"
    if d < start_this:
        # Aún no arranca esta temporada en este año.
        return f"{d.year}_{d.year}"
    # Después del end: la próxima es el siguiente año.
    return f"{d.year + 1}_{d.year + 1}"
