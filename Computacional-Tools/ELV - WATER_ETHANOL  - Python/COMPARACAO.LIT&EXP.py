 
import matplotlib.pyplot as plt

# ===============================
# DADOS DA LITERATURA
# Rieder & Thompson (1949)
# ===============================
dados_literatura = [
    {"T_C": 99.30, "x1": 0.0028, "y1": 0.0320},
    {"T_C": 96.90, "x1": 0.0118, "y1": 0.1130},
    {"T_C": 96.00, "x1": 0.0137, "y1": 0.1570},
    {"T_C": 95.60, "x1": 0.0176, "y1": 0.1560},
    {"T_C": 94.80, "x1": 0.0222, "y1": 0.1860},
    {"T_C": 93.80, "x1": 0.0246, "y1": 0.2120},
    {"T_C": 92.90, "x1": 0.0331, "y1": 0.2480},
    {"T_C": 90.50, "x1": 0.0519, "y1": 0.3180},
    {"T_C": 89.40, "x1": 0.0625, "y1": 0.3390},
    {"T_C": 87.20, "x1": 0.0871, "y1": 0.4600},
    {"T_C": 85.40, "x1": 0.1260, "y1": 0.4680},
    {"T_C": 84.50, "x1": 0.1430, "y1": 0.4870},
    {"T_C": 83.40, "x1": 0.1720, "y1": 0.5050},
    {"T_C": 82.30, "x1": 0.2550, "y1": 0.5520},
    {"T_C": 81.20, "x1": 0.3450, "y1": 0.5910},
    {"T_C": 80.50, "x1": 0.4500, "y1": 0.6330},
    {"T_C": 80.00, "x1": 0.5960, "y1": 0.6610},
    {"T_C": 79.50, "x1": 0.5450, "y1": 0.6730},
    {"T_C": 78.80, "x1": 0.6630, "y1": 0.7310},
    {"T_C": 78.50, "x1": 0.7350, "y1": 0.7760},
    {"T_C": 78.40, "x1": 0.8040, "y1": 0.8150},
    {"T_C": 78.30, "x1": 0.9170, "y1": 0.9060},
]

# ===============================
# DADOS EXPERIMENTAIS DO GRUPO
# ===============================
dados_exp = [
    {"T_C": 99.2, "x1": 0.0000, "y1": 0.0000},
    {"T_C": 94.7, "x1": 0.0000, "y1": 0.1613},
    {"T_C": 89.7, "x1": 0.0146, "y1": 0.2716},
    {"T_C": 80.7, "x1": 0.2111, "y1": 0.5281},
    {"T_C": 79.2, "x1": 0.4600, "y1": 0.6397},
    {"T_C": 78.0, "x1": 0.7902, "y1": 0.8009},
]

# ===============================
# ORGANIZAR DADOS
# ===============================

x_lit = [p["x1"] for p in dados_literatura]
y_lit = [p["y1"] for p in dados_literatura]
T_lit = [p["T_C"] for p in dados_literatura]

x_exp = [p["x1"] for p in dados_exp]
y_exp = [p["y1"] for p in dados_exp]
T_exp = [p["T_C"] for p in dados_exp]

# ===============================
# GRÁFICO
# ===============================

plt.figure(figsize=(12, 7))

# Literatura - líquido
plt.plot(
    x_lit, T_lit,
    marker="o",
    linestyle="-",
    linewidth=1.8,
    markersize=7,
    markerfacecolor="none",
    label="Líquido literatura"
)

# Literatura - vapor
plt.plot(
    y_lit, T_lit,
    marker="o",
    linestyle="-",
    linewidth=1.8,
    markersize=7,
    markerfacecolor="none",
    label="Vapor literatura"
)

# Experimental - líquido
plt.plot(
    x_exp, T_exp,
    marker="o",
    linestyle="-",
    linewidth=1.8,
    markersize=7,
    markerfacecolor="none",
    label="Líquido experimento"
)

# Experimental - vapor
plt.plot(
    y_exp, T_exp,
    marker="o",
    linestyle="-",
    linewidth=1.8,
    markersize=7,
    markerfacecolor="none",
    label="Vapor experimento"
)
plt.xlabel("Fração molar de etanol", fontsize=16)
plt.ylabel("Temperatura (°C)", fontsize=16)

plt.title(
    "Comparação T-x-y água-etanol — Literatura vs Experimental",
    fontsize=18
)

plt.xlim(0, 1)
plt.ylim(77, 101)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True, fontsize=13)

plt.tight_layout()

plt.savefig(
    "comparacao_Txy_literatura_vs_experimental.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()