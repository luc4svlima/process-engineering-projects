import matplotlib.pyplot as plt

# Dados experimentais do Excel
x_egme = [0.0169, 0.0202, 0.0365, 0.0614, 0.0908, 0.1518, 0.1664]

T_media_C = [59.15, 50.25, 41.75, 41.30, 46.30, 53.95, 60.50]

exp = [1, 2, 3, 4, 5, 6, 7]

plt.figure(figsize=(8, 6))

plt.plot(x_egme, T_media_C, marker="o", linestyle="-", label="Dados experimentais")

for i, x, T in zip(exp, x_egme, T_media_C):
    plt.text(x, T + 0.4, str(i), fontsize=9)

plt.xlabel("Fração molar de EGME")
plt.ylabel("Temperatura média (°C)")
plt.title("Diagrama experimental ELL - EGME + água")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_experimental_ELL.png", dpi=300)
plt.show()