import pandas as pd
from datetime import datetime

from cicese import read_cicese_data


# PENDIENTE: hacer formato bien, hacer loop por todas las estaciones

def main():

    directory_from = "C:\\Users\\javi2\\Documents\\CD_aplicada_1\\COBI\\etl\\data\\cicese\\raw\\"
    directory_to = "C:\\Users\\javi2\\Documents\\CD_aplicada_1\\COBI\\etl\\data\\cicese\\processed\\"

    estacion = "guerrero_negro"
    df = read_cicese_data(estacion, directory_from)
    df = df.groupby(["anio", "mes", "dia"]).median()
    df["estacion"] = estacion
    df = df.reset_index()
    df["mes"] = df.apply(lambda x: "0"+str(x["mes"]) if x["mes"] < 10 else x["mes"], axis=1)
    df["dia"] = df.apply(lambda x: "0"+str(x["dia"]) if x["dia"] < 10 else x["dia"], axis=1)
    df["date"] = df.apply(lambda x: str(x["anio"])+"-"+str(x["mes"])+"-"+str(x["dia"]), axis=1)

    df = df[["estacion", "nivel_mar_leveltrol", "nivel_mar_burbujeador", "nivel_mar_ott_rsl", "nivel_mar_sutron", 
            "temperatura_agua", "radiacion_solar","direccion_viento","magnitud_viento","temperatura_aire",
            "humedad_relativa","presion_atmosferica","precipitacion", "date"]]

    df.to_csv(directory_to+estacion+".csv")

if __name__ == '__main__':
    main()
            