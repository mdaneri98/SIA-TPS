import csv
import numpy as np
import matplotlib.pyplot as plt

# Leer los datos del archivo CSV y guardar el rendimiento por generación para cada archivo
def leer_datos(archivo_csv):
    generaciones = []
    promedios_performance = []
    desvios_performance = []

    with open(archivo_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            generaciones.append(int(row[0]))
            promedios_performance.append(float(row[2]))
            desvios_performance.append(float(row[3]))

    return generaciones, promedios_performance, desvios_performance

# Archivos CSV
archivos_csv = ['datos_Arquero1.csv', 'datos_Arquero2.csv']

# Graficar promedio de rendimiento con intervalo de confianza para cada archivo
for archivo_csv in archivos_csv:
    generaciones, promedio_performance, desvio_performance = leer_datos(archivo_csv)
    plt.plot(generaciones, promedio_performance, linestyle='-')
    plt.fill_between(generaciones, np.array(promedio_performance) - np.array(desvio_performance),
                     np.array(promedio_performance) + np.array(desvio_performance), alpha=0.3)

plt.xlabel('Generación')
plt.ylabel('Promedio de Performance')
plt.title('Promedio de Performance por Generación con Intervalo de Confianza')
plt.grid(True)
plt.legend(['datos_Arquero1.csv', 'datos_Arquero2.csv'])

plt.show()