import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def biplot(score, coeff, countries, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    plt.figure(figsize=(14, 8))
    plt.scatter(xs, ys, s=50)

    for i, country in enumerate(countries):
        plt.annotate(country, (xs[i], ys[i]))

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


def plot_pca_index_by_country(countries, principal_df):
    plt.figure(figsize=(14, 8))
    plt.bar(countries, principal_df["PC1"])
    plt.xlabel('Países')
    plt.ylabel('Índice de la Primera Componente Principal (PC1)')
    plt.title('Índice de la Primera Componente Principal según País')
    plt.xticks(rotation=90)
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

countries = df["Country"]

# Biplot para la primera componente
biplot(pca_result, pca.components_.T, countries, labels=variables)

countries = df['Country']
df.drop(columns=['Country'])

pca = PCA(n_components=2)

principal_components = pca.fit_transform(df_scaled[variables])
principal_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

print(principal_df)

plot_pca_index_by_country(countries, principal_df)


# Interpretación de la primera componente
for i in range(0, len(explained_variance)):
    print(f"Varianza explicada por componente {i}: {explained_variance[i]:.2f}")

# Componentes de la primera PC
first_pc = pca.components_[0]
print("Componentes de la primera PC:")
for i, var in enumerate(variables):
    print(f"{var}: {first_pc[i]:.4f}")


