import numpy as np
import matplotlib.pyplot as plt
from math import exp, log
from scipy.optimize import root

# =========================
# DADOS EXPERIMENTAIS
# =========================

x_exp = np.array([0.0169, 0.0202, 0.0365, 0.0614, 0.0908, 0.1518, 0.1664])
T_exp_C = np.array([59.15, 50.25, 41.75, 41.30, 46.30, 53.95, 60.50])
T_exp_K = T_exp_C + 273.15

# =========================
# CONSTANTES
# =========================

R = 8.3144598
alfa = 0.2

# =========================
# NRTL
# =========================

def tau12(A12, T): return A12 / (R * T)
def tau21(A21, T): return A21 / (R * T)

def G12(A12, T): return exp(-alfa * tau12(A12, T))
def G21(A21, T): return exp(-alfa * tau21(A21, T))

def gamma1(x1, A12, A21, T):
    x2 = 1 - x1
    t12, t21 = tau12(A12, T), tau21(A21, T)
    g12, g21 = G12(A12, T), G21(A21, T)

    return exp(x2**2 * (
        t21 * (g21 / (x1 + x2 * g21))**2 +
        (t12 * g12) / (x2 + x1 * g12)**2
    ))

def gamma2(x1, A12, A21, T):
    x2 = 1 - x1
    t12, t21 = tau12(A12, T), tau21(A21, T)
    g12, g21 = G12(A12, T), G21(A21, T)

    return exp(x1**2 * (
        t12 * (g12 / (x2 + x1 * g12))**2 +
        (t21 * g21) / (x1 + x2 * g21)**2
    ))

# =========================
# AJUSTE DE PARÂMETROS
# =========================

def fobj_parametros(A, xa, xb, T):
    A12, A21 = A

    return [
        log(xa * gamma1(xa, A12, A21, T)) - log(xb * gamma1(xb, A12, A21, T)),
        log((1 - xa) * gamma2(xa, A12, A21, T)) - log((1 - xb) * gamma2(xb, A12, A21, T))
    ]

pares = [
    (x_exp[0], x_exp[6], (T_exp_K[0] + T_exp_K[6]) / 2),
    (x_exp[1], x_exp[5], (T_exp_K[1] + T_exp_K[5]) / 2),
    (x_exp[2], x_exp[4], (T_exp_K[2] + T_exp_K[4]) / 2),
]

A12_lista, A21_lista, T_pares = [], [], []

for xa, xb, T in pares:
    sol = root(fobj_parametros, [-17000, 34000], args=(xa, xb, T))

    A12_lista.append(sol.x[0])
    A21_lista.append(sol.x[1])
    T_pares.append(T)

# regressão linear
coef_A12 = np.polyfit(T_pares, A12_lista, 1)
coef_A21 = np.polyfit(T_pares, A21_lista, 1)

def A12_T(T): return coef_A12[0] * T + coef_A12[1]
def A21_T(T): return coef_A21[0] * T + coef_A21[1]

# =========================
# MODELO BINODAL (com CONTINUIDADE REAL)
# =========================

def fobj_modelagem(X, T, fator):
    xa, xb = X

    A12 = A12_T(T) * fator
    A21 = A21_T(T) * fator

    return [
        xa * gamma1(xa, A12, A21, T) - xb * gamma1(xb, A12, A21, T),
        (1 - xa) * gamma2(xa, A12, A21, T) - (1 - xb) * gamma2(xb, A12, A21, T)
    ]

# =========================
# BINODAL COM CONTINUIDADE (FIX REAL)
# =========================

def calcular_binodal(fator):
    T_vals, xa_vals, xb_vals = [], [], []

    chute = [x_exp[0], x_exp[-1]]
    prev_xa, prev_xb = chute

    for T in np.arange(max(T_pares), 313.0, -0.1):

        sol = root(fobj_modelagem, chute, args=(T, fator))

        if sol.success:
            xa, xb = sol.x

            # restrições físicas
            if 0 < xa < xb < 1 and (xb - xa) > 0.02:

                # 🔥 CONTINUIDADE (ESSENCIAL PARA NÃO QUEBRAR CURVA)
                if abs(xa - prev_xa) > 0.05 or abs(xb - prev_xb) > 0.05:
                    continue

                T_vals.append(T)
                xa_vals.append(xa)
                xb_vals.append(xb)

                prev_xa, prev_xb = xa, xb
                chute = [xa, xb]

    return np.array(T_vals), np.array(xa_vals), np.array(xb_vals)

# =========================
# GRÁFICO FINAL (NÍVEL PAPER)
# =========================

plt.figure(figsize=(10, 7))

casos = [
    (1.0, "Base"),
    (0.9, "-10%"),
    (1.1, "+10%")
]

for fator, label in casos:

    T_mod, xa_mod, xb_mod = calcular_binodal(fator)

    if len(T_mod) < 5:
        continue

    T_C = T_mod - 273.15

    # ordenação final (segurança visual)
    idx = np.argsort(T_C)
    T_C = T_C[idx]
    xa_mod = xa_mod[idx]
    xb_mod = xb_mod[idx]

    # curva conectada (sem spline agressiva)
    plt.plot(xa_mod, T_C, linewidth=2.2, label=label)
    plt.plot(xb_mod, T_C, linewidth=2.2)

# experimental
plt.scatter(
    x_exp,
    T_exp_C,
    color="black",
    s=60,
    label="Dados experimentais",
    zorder=5
)

# estilo paper
plt.xlabel("Fração molar de EGME")
plt.ylabel("Temperatura (°C)")
plt.title("Análise de sensibilidade dos parâmetros NRTL")

plt.grid(True, linestyle=":", alpha=0.4)
plt.legend()

plt.xlim(0.0, 0.19)
plt.ylim(38, 63)

plt.tight_layout()
plt.savefig("NRTL_paper_final.png", dpi=300, bbox_inches="tight")
plt.show()