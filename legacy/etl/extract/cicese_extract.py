from cicese import gather_cicese_data
import os

def main():

    directory_to = os.path.join(os.getcwd(),"data","cicese", "raw")

    if not os.path.isdir(directory_to):
        os.makedirs(directory_to, exist_ok=True)

    # PENDIENTE: pasar a limpio, hacer un loop por todas las estaciones

    gather_cicese_data(2021, directory_to=directory_to, location="guerrero_negro")
    # gather_cicese_data(2021, directory_to=directory_to, location="isla_cedros")

if __name__ == '__main__':
    main()
