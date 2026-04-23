import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/digni/OneDrive/Documents/GitHub/Tesis/dataset/delitos_total.csv')

# Contar SI / NO
conteo = df['uso_arma'].value_counts()
conteo = conteo[['SI','NO']]  # asegurar orden

# Calcular porcentajes
porcentajes = conteo / conteo.sum() * 100

plt.figure(figsize=(6,5))
ax = sns.barplot(x=conteo.index, y=conteo.values, palette="coolwarm")

# Agregar los porcentajes encima de cada barra
for i, (valor, pct) in enumerate(zip(conteo.values, porcentajes)):
    ax.text(i, valor + valor*0.01, f"{pct:.1f}%", ha='center', va='bottom', fontsize=12)

plt.title('Uso de arma en delitos (SI / NO)', fontsize=15)
plt.xlabel('Uso de arma')
plt.ylabel('Cantidad de delitos')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
