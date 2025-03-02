import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.offsetbox as offsetbox
import numpy as np
import statsmodels.api as sm
import matplotlib.colors as mcolors
import math

# 📂 Caminhos dos arquivos
caminho_csv = r"C:\Users\miali\Downloads\Agora sou cientista e ninguém vai me segurar\pokemon\pokemons.csv"
caminho_imagens = r"C:\Users\miali\Downloads\Agora sou cientista e ninguém vai me segurar\pokemon\imagens"

# 📊 Carregar os dados e remover duplicatas
df = pd.read_csv(caminho_csv)
df = df.drop_duplicates(subset=["Nome"]).reset_index(drop=True)

# Se não houver coluna "Grupo", defina todos como "A"
if "Grupo" not in df.columns:
    df["Grupo"] = "A"

# 🎨 Definir uma paleta em tons pastéis para o gradiente harmônico
correcao_cores = {
    "Preto": "#2E2E2E",
    "Cinza": "#D6D6D6",
    "Branco": "#F8F8F8",  
    "Marrom": "#C69C6D",
    "Vermelho": "#FFA6A6",
    "Laranja": "#FFC999",
    "Amarelo": "#FFEE99",
    "Verde": "#A8E6A1",
    "Azul": "#A1C8E6",
    "Roxo": "#D1A3E6",
    "Rosa": "#F4C2C2"
}
df["Cor Corrigida"] = df["Cor"].map(correcao_cores)

# ─── Ordenar os dados conforme a paleta (agrupando cores similares próximas)
ordem_custom = ["Preto", "Cinza", "Branco", "Marrom", "Rosa", "Vermelho", "Laranja", "Amarelo", "Verde", "Azul", "Roxo"]
df["Ordem"] = df["Cor"].apply(lambda x: ordem_custom.index(x))
df = df.sort_values("Ordem").reset_index(drop=True)

# ─── Distribuir os pontos com jitter reduzido
num_pokemons = len(df)
df["X"] = np.linspace(1, 15, num_pokemons) + np.random.uniform(-0.2, 0.2, num_pokemons)
df["Y"] = df["Ordem"] * 2.0 + np.random.uniform(-0.3, 0.3, num_pokemons)

# ─── Ajustar tamanhos: Pokémon grandes ficam maiores
pokemons_grandes = ["Snorlax", "Charizard", "Blastoise", "Venusaur", "Dragonite", "Gyarados"]
df["Tamanho"] = df["Nome"].apply(lambda x: 55 if x in pokemons_grandes else 40)

# ─── Calcular regressão linear para cada cor e ajustar Y para "costurar" a linha com efeito senoidal
df["Y_adj"] = df["Y"].copy()
for cor, group_df in df.groupby("Cor"):
    group_df = group_df.sort_values("X")
    X_grp = sm.add_constant(group_df["X"])
    model_grp = sm.OLS(group_df["Y"], X_grp).fit()
    predicted_grp = model_grp.predict(X_grp)
    delta = 1.0  # valor de desvio para separar mais os pontos da mesma cor
    offsets = np.array([delta if i % 2 == 0 else -delta for i in range(len(group_df))])
    df.loc[group_df.index, "Y_adj"] = predicted_grp + offsets

# ─── REPULSÃO: Evitar sobreposição entre os Pokémon
df["X_final"] = df["X"].copy()
df["Y_final"] = df["Y_adj"].copy()

min_dist = 1.0   # Distância mínima entre pontos
n_iterations = 10  # Número de iterações de repulsão
learning_rate = 0.5  # Força do empurrão

for _ in range(n_iterations):
    moved = False
    for i in range(num_pokemons):
        for j in range(i+1, num_pokemons):
            dx = df.loc[j, "X_final"] - df.loc[i, "X_final"]
            dy = df.loc[j, "Y_final"] - df.loc[i, "Y_final"]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                overlap = (min_dist - dist) / 2.0
                if dist == 0:
                    angle = np.random.uniform(0, 2*np.pi)
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                    dist = 1
                nx = dx / dist
                ny = dy / dist
                df.loc[i, "X_final"] -= learning_rate * overlap * nx
                df.loc[i, "Y_final"] -= learning_rate * overlap * ny
                df.loc[j, "X_final"] += learning_rate * overlap * nx
                df.loc[j, "Y_final"] += learning_rate * overlap * ny
                moved = True
    if not moved:
        break

# ─── Configurar visualização do scatter plot
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(12, 8))

# Plot base opcional (com baixa opacidade)
sns.scatterplot(
    x=df["X_final"],
    y=df["Y_final"],
    hue=df["Grupo"],
    palette={"A": "#B5B5FF", "B": "#FFB5E6"},
    alpha=0.3,
    s=100,
    ax=ax
)

# Plotar a linha de tendência global (com alpha=0 para invisibilidade)
X_all = sm.add_constant(df["X"])
model_all = sm.OLS(df["Y"], X_all).fit()
x_line = np.linspace(df["X"].min(), df["X"].max(), 100)
y_line = model_all.predict(sm.add_constant(x_line))
pastel_line_color = mcolors.to_rgba("#B0E0E6", alpha=0)
ax.plot(x_line, y_line, color=pastel_line_color, linewidth=2.5)

# ─── Função para criar textura de Pokémon com fundo suave (borda com mais cor, mas mantendo transparência)
def criar_textura_pokemon(imagem, cor_fundo, tamanho=50, blur=2):
    """
    Redimensiona a imagem para o tamanho desejado e cria um fundo sutil com
    uma leve cor de fundo (transparente) para substituir a borda escura.
    Agora, a opacidade do fundo foi aumentada de "10" para "40" para ampliar a sensação do grupo.
    """
    img = imagem.resize((tamanho, tamanho), Image.LANCZOS)
    fundo = Image.new("RGBA", (tamanho + blur*2, tamanho + blur*2), (0, 0, 0, 0))
    mask = Image.new("L", fundo.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((blur, blur, tamanho + blur, tamanho + blur), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    background = Image.new("RGBA", fundo.size, cor_fundo + "60")  # opacidade aumentada para "60"
    background.paste(img, (blur, blur), img)
    fundo.paste(background, (0, 0), mask)
    return fundo

# ─── Listar arquivos de imagens disponíveis
arquivos_disponiveis = os.listdir(caminho_imagens)

# ─── Adicionar as imagens dos Pokémon como textura nos pontos do scatter plot
for i, row in df.iterrows():
    try:
        nome_formatado = row["Nome"].lower().replace(" ", "").replace("-", "")
        arquivo_pokemon = next((arq for arq in arquivos_disponiveis 
                                if nome_formatado in arq.lower().replace(" ", "").replace("-", "")), None)
        if arquivo_pokemon:
            caminho_imagem = os.path.join(caminho_imagens, arquivo_pokemon)
            img = Image.open(caminho_imagens + "\\" + arquivo_pokemon).convert("RGBA")
            textura = criar_textura_pokemon(img, correcao_cores[row["Cor"]], tamanho=row["Tamanho"])
            img_box = offsetbox.OffsetImage(textura, zoom=0.75)
            img_ab = offsetbox.AnnotationBbox(img_box, (row["X_final"], row["Y_final"]), frameon=False)
            ax.add_artist(img_ab)
    except Exception as e:
        print(f"❌ ERRO ao adicionar imagem para {row['Nome']}: {e}")

# ─── Ajustar layout do gráfico
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")  # Rótulo removido
ax.set_ylabel("Resultado")
plt.title("Gráfico de Dispersão dos Pokémon (Pastel, Fundo Suave, Linha Transparente)", fontsize=14)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
print(f"Total de Pokémon no gráfico: {len(df)}")
plt.show()