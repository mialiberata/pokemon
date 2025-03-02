import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do CSV
df = pd.read_csv("pokemons.csv")

# Contar quantos Pokémons existem de cada cor
contagem_cores = df["Cor"].value_counts()

# Criar uma lista de cores baseadas nos nomes em inglês da PokeAPI
cores_mapeadas = {
    "red": "red",
    "blue": "blue",
    "yellow": "yellow",
    "green": "green",
    "black": "black",
    "brown": "saddlebrown",
    "purple": "purple",
    "gray": "gray",
    "white": "white",
    "pink": "pink"
}

# Criar um gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(contagem_cores.index, contagem_cores.values, color=[cores_mapeadas[c] for c in contagem_cores.index])

# Personalizar o gráfico
plt.xlabel("Cor do Pokémon")
plt.ylabel("Quantidade")
plt.title("Distribuição de Cores dos Pokémons")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Mostrar o gráfico (sem salvar automaticamente)
plt.show()
