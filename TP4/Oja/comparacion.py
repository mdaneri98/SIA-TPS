import numpy as np
import matplotlib.pyplot as plt

# Etiquetas para las variables
labels = ['Area', 'GDP', 'Inflation', 'Exp. de vida', 'Militares', 'Inc. de población', 'Desempleo']

# Componentes principales de Oja
oja_principal_components_1000 = np.array([-0.20275538, 0.48993905, -0.47946847, 0.50095825, -0.13074739, 0.45684623, -0.232006])

# Componentes principales de otro método (supongo que de scikit-learn)
sci_principal_components = np.array([0.12, -0.50, -0.4065, -0.4829, 0.1881, -0.4757, 0.2717])

# Convertir todas las componentes a valores absolutos (positivos)
oja_principal_components_positive = np.abs(oja_principal_components_1000)
sci_principal_components_positive = np.abs(sci_principal_components)

# Calcular la diferencia entre las componentes principales positivas
differences = oja_principal_components_positive - sci_principal_components_positive

# Crear el gráfico de barras
x = np.arange(len(labels))  # El rango de las posiciones en el eje x
width = 0.35  # El ancho de las barras

fig, ax = plt.subplots(figsize=(14, 8))

# Barras para las componentes principales de Oja
rects1 = ax.bar(x - width/2, oja_principal_components_positive, width, label='Oja (Positivo)')

# Barras para las componentes principales del otro método
rects2 = ax.bar(x + width/2, sci_principal_components_positive, width, label='Otro método (Positivo)')

# Añadir etiquetas, título y leyenda
ax.set_ylabel('Valor de las cargas')
ax.set_xlabel('Característica asociada')
ax.set_title('Comparación de las cargas de la comp. principal de Oja y la librería')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Añadir las diferencias
for i in range(len(x)):
    ax.text(i - width/2, oja_principal_components_positive[i], f'{oja_principal_components_positive[i]:.3f}', ha='center', va='bottom')
    ax.text(i + width/2, sci_principal_components_positive[i], f'{sci_principal_components_positive[i]:.3f}', ha='center', va='bottom')

# Mostrar el gráfico
plt.show()

# Crear un gráfico de barras separado para las diferencias
differences = np.abs(differences)
fig, ax = plt.subplots(figsize=(14, 8))
rects3 = ax.bar(x, differences, width, label='Diferencia (Oja - Librería)')

# Añadir etiquetas, título y leyenda
ax.set_ylabel('Diferencia de las cargas')
ax.set_xlabel('Característica asociada')
ax.set_title('Diferencia de las cargas de la comp. principal de Oja y la librería')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Añadir las diferencias
for i in range(len(x)):
    ax.text(i, differences[i], f'{differences[i]:.4f}', ha='center', va='bottom')

# Mostrar el gráfico
plt.show()
