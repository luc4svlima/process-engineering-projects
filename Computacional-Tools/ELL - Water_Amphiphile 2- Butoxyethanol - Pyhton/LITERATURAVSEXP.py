import numpy as np
import matplotlib.pyplot as plt

# =========================
# Dados experimentais do grupo
# =========================

x_exp = np.array([0.0169, 0.0202, 0.0365, 0.0614, 0.0908, 0.1518, 0.1664])
T_exp_k = np.array([332.3, 323.4, 314.9, 314.45, 319.45, 327.10, 333.65])

# =========================
# Literatura Kim & Lim, 2001
# Dados em fração mássica de EGME
# =========================

T_lit_K = np.array([
    323.18, 324.19, 324.92, 326.12, 327.14,
    328.07, 329.17, 330.13, 331.14, 333.10,
    335.17, 337.01, 338.99, 341.00, 342.94
])

w_alpha = np.array([
    0.1826, 0.1667, 0.1529, 0.1469, 0.1400,
    0.1360, 0.1340, 0.1236, 0.1176, 0.1188,
    0.1152, 0.1086, 0.1094, 0.1048, 0.1071
])

w_beta = np.array([
    0.4211, 0.4484, 0.4694, 0.4752, 0.4977,
    0.5140, 0.5155, 0.5452, 0.5525, 0.5533,
    0.5630, 0.5834, 0.5896, 0.5958, 0.6121
])

#T_lit_C = T_lit_K - 273.15

# =========================
# Conversão w -> x
# =========================

MM_EGME = 118.176
MM_H2O = 18.015

def w_to_x_egme(w):
    n_egme = w / MM_EGME
    n_agua = (1 - w) / MM_H2O
    return n_egme / (n_egme + n_agua)

x_alpha_lit = w_to_x_egme(w_alpha)
x_beta_lit = w_to_x_egme(w_beta)

# =========================
# Gráfico comparação experimental x literatura
# =========================

plt.figure(figsize=(10, 7))


plt.scatter(
    x_exp,
    T_exp_k,
    #T_lit_C,
    color="black",
    marker="o",
    s=65,
    label="Experimental",
    zorder=5
)

plt.scatter(
    x_alpha_lit,
    T_lit_K,
    #T_lit_C,
    marker="x",
    s=55,
    label="Literatura - fase α"
)

plt.scatter(
    x_beta_lit,
    T_lit_K,
    #T_lit_C,
    marker="x",
    s=55,
    label="Literatura - fase β"
)

for i, (x, T) in enumerate(zip(x_exp, T_exp_k), start=1):
    plt.annotate(
        str(i),
        (x, T),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9
    )

plt.xlabel("Fração molar de EGME", fontsize=12)
plt.ylabel("Temperatura (k)", fontsize=12)
plt.title("Literatura vs Experimental", fontsize=14)

plt.grid(False)
plt.legend(fontsize=11)

plt.ylim(min(T_exp_k.min(), T_lit_K.min()) - 2, max(T_exp_k.max(), T_lit_K.max()) + 2)
plt.xlim(0, max(x_beta_lit.max(), x_exp.max()) + 0.02)

plt.tight_layout()
plt.savefig("comparacao_experimental_literatura.png", dpi=300, bbox_inches="tight")
plt.show()