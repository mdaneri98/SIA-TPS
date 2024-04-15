import tkinter as tk
from tkinter import ttk
import csv
import os

def analizar_content(archivo):
    with open(archivo, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[2] == 'content':
                return row[0], row[2]
    return None

def analizar_optimal(archivo):
    with open(archivo, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[2] == 'optimal':
                return row[0], row[2]
    return None 

def comparar_archivos(archivos):
    tabla = []
    for archivo in archivos:
        nombre_archivo, _ = os.path.splitext(os.path.basename(archivo))
        analisis1 = analizar_content(archivo)
        analisis2 = analizar_optimal(archivo)
        if analisis1:
            tabla.append((nombre_archivo, analisis1[0], analisis1[1]))
        if analisis2:
            tabla.append((nombre_archivo, analisis2[0], analisis2[1]))
    return tabla

def mostrar_tabla():
    ventana = tk.Tk()
    ventana.title("Comparación de Archivos")

    # Crear el Treeview
    tree = ttk.Treeview(ventana, columns=("Archivo", "Nro Generación", "Corte"), show="headings")
    tree.heading("Archivo", text="Archivo")
    tree.heading("Nro Generación", text="Nro Generación")
    tree.heading("Corte", text="Corte")
    
    tabla = comparar_archivos(["elitista.csv", "ranking.csv",
                               "ruleta.csv", "universal.csv",
                               "boltzmann.csv", "torneo_deterministico.csv",
                                "torneo_probabilistico.csv"])

    for fila in tabla:
        tree.insert("", "end", values=fila)

    tree.pack(expand=True, fill="both")

    ventana.mainloop()

mostrar_tabla()
