import pandas as pd
import psycopg2

# PENDIENTE: hacer formato bien, hacer loop por todas las estaciones
def main():

    directory_to = "C:\\Users\\javi2\\Documents\\CD_aplicada_1\\COBI\\etl\\data\\cicese\\processed\\"

    estacion = "guerrero_negro"
    df = pd.read_csv(directory_to+estacion+".csv")

    try:

        # set up a connection to the database
        conn = psycopg2.connect(
            host="localhost",
            database="cobi",
            user="postgres",
            password="admin"
        )

        # create a database cursor
        cur = conn.cursor()


        # Iterate over the DataFrame rows and insert them into the PostgreSQL table
        for i, row in df.iterrows():
            cur.execute("""INSERT INTO cicese (estacion, nivel_mar_leveltrol, nivel_mar_burbujeador, nivel_mar_ott_rsl, 
            nivel_mar_sutron, temperatura_agua, radiacion_solar,direccion_viento,magnitud_viento,temperatura_aire,
            humedad_relativa,presion_atmosferica,precipitacion,date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (row["estacion"], row["nivel_mar_leveltrol"], row["nivel_mar_burbujeador"], row["nivel_mar_ott_rsl"], 
                        row["nivel_mar_sutron"], row["temperatura_agua"], row["radiacion_solar"], row["direccion_viento"],
                        row["magnitud_viento"], row["temperatura_aire"], row["humedad_relativa"], row["presion_atmosferica"],
                        row["precipitacion"], row["date"]))

        # Commit the changes and close the database connection
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print(" error:", e)

if __name__ == '__main__':
    main()
            