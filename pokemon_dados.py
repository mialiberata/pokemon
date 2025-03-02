import pandas as pd

# Caminho do arquivo CSV
caminho_csv = r"C:\Users\miali\Downloads\Agora sou cientista e ninguém vai me segurar\pokemon\pokemons.csv"

# Carregar os dados do CSV
try:
    df = pd.read_csv(caminho_csv, dtype=str)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Remove espaços extras
    print("\n📊 Tabela de Pokémons:")
    print(df.head(10))  # Mostra as 10 primeiras linhas no terminal
except Exception as e:
    print(f"❌ Erro ao carregar CSV para exibição: {e}")
