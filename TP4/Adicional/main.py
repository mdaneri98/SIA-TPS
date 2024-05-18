import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

variables = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']


def normalize_data(data_frame):
    df_normalizado = data_frame.copy()
    for variable in variables:
        media = data_frame[variable].mean()
        desviacion_estandar = data_frame[variable].std()
        df_normalizado[variable] = (data_frame[variable] - media) / desviacion_estandar
    return df_normalizado


def graph_box_plot(data_frame):
    data_frame[variables].boxplot(figsize=(12, 8))
    plt.title('Boxplot de las variables')
    plt.xticks(rotation=45)
    plt.show()


def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    plt.figure(figsize=(14, 8))
    plt.scatter(xs, ys, s=50)

    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0] * 2, coeff[i, 1] * 2, color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 2.5, coeff[i, 1] * 2.5, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 2.5, coeff[i, 1] * 2.5, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.title('Biplot de las dos primeras componentes principales')
    plt.show()


def plot_component_loadings(components, number):
    plt.figure(figsize=(10, 6))
    plt.bar(variables, components[number])
    plt.xlabel('Variables')
    plt.ylabel('Cargas')
    plt.title(f'Cargas de la Componente Principal {number+1}')
    plt.xticks(rotation=45)
    plt.show()


# Cargar el conjunto de datos
df = pd.read_csv('europe.csv')

# Graph raw data
graph_box_plot(df)

# Normalize
df_scaled = normalize_data(df)

# Graph normalized data
graph_box_plot(df_scaled)

# Aplicación del PCA
pca = PCA()
pca_result = pca.fit_transform(df_scaled[variables])

# Varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_

# Biplot para la primera componente
biplot(pca_result, pca.components_.T, labels=variables)

plot_component_loadings(pca.components_, number=0)


# Interpretación de la primera componente
print(f"Varianza explicada por la primera componente: {explained_variance[0]:.2f}")

# Componentes de la primera PC
first_pc = pca.components_[0]
print("Componentes de la primera PC:")
for i, var in enumerate(variables):
    print(f"{var}: {first_pc[i]:.4f}")


'''
    Como la carga de la 'area' no es significativamente grande respecto de las demás, podemos decir que 

'''


