import numpy as np
import matplotlib.pyplot as plt
from math import exp, log
from scipy.optimize import root

# =========================
# Dados experimentais
# =========================

x_exp = np.array([0.0169, 0.0202, 0.0365, 0.0614, 0.0908, 0.1518, 0.1664])
T_exp_C = np.array([59.15, 50.25, 41.75, 41.30, 46.30, 53.95, 60.50])
T_exp_K = T_exp_C + 273.15

R = 8.3144598
alfa = 0.2

# =========================
# Funções NRTL
# =========================

def tau12(A12, T):
    return A12 / (R * T)

def tau21(A21, T):
    return A21 / (R * T)

def G12(A12, T):
    return exp(-alfa * tau12(A12, T))

def G21(A21, T):
    return exp(-alfa * tau21(A21, T))

def gamma1(x1, A12, A21, T):
    x2 = 1 - x1
    t12 = tau12(A12, T)
    t21 = tau21(A21, T)
    g12 = G12(A12, T)
    g21 = G21(A21, T)

    ln_gamma = x2**2 * (
        t21 * (g21 / (x1 + x2 * g21))**2
        + (t12 * g12) / (x2 + x1 * g12)**2
    )
    return exp(ln_gamma)

def gamma2(x1, A12, A21, T):
    x2 = 1 - x1
    t12 = tau12(A12, T)
    t21 = tau21(A21, T)
    g12 = G12(A12, T)
    g21 = G21(A21, T)

    ln_gamma = x1**2 * (
        t12 * (g12 / (x2 + x1 * g12))**2
        + (t21 * g21) / (x1 + x2 * g21)**2
    )
    return exp(ln_gamma)

# =========================
# Ajuste dos parâmetros A12 e A21
# =========================

def fobj_parametros(A, x_alpha, x_beta, T):
    A12, A21 = A

    eq1 = log(x_alpha * gamma1(x_alpha, A12, A21, T)) - log(
        x_beta * gamma1(x_beta, A12, A21, T)
    )

    eq2 = log((1 - x_alpha) * gamma2(x_alpha, A12, A21, T)) - log(
        (1 - x_beta) * gamma2(x_beta, A12, A21, T)
    )

    return [eq1, eq2]

pares = [
    ("1-7", x_exp[0], x_exp[6], (T_exp_K[0] + T_exp_K[6]) / 2),
    ("2-6", x_exp[1], x_exp[5], (T_exp_K[1] + T_exp_K[5]) / 2),
    ("3-5", x_exp[2], x_exp[4], (T_exp_K[2] + T_exp_K[4]) / 2),
]

chute_A = [-17000, 34000]

A12_lista = []
A21_lista = []
T_pares = []

print("\nTabela de parâmetros NRTL")
print("-" * 70)
print("Par     T(K)       T(°C)      A12          A21")
print("-" * 70)

for nome_par, x_alpha, x_beta, T in pares:
    sol = root(fobj_parametros, chute_A, args=(x_alpha, x_beta, T))

    if not sol.success:
        print(f"Atenção: o ajuste do par {nome_par} não convergiu.")

    A12, A21 = sol.x

    A12_lista.append(A12)
    A21_lista.append(A21)
    T_pares.append(T)

    print(
        f"{nome_par:<6} "
        f"{T:>8.2f}   "
        f"{T - 273.15:>8.2f}   "
        f"{A12:>10.2f}   "
        f"{A21:>10.2f}"
    )

# =========================
# Regressão linear A12(T) e A21(T)
# =========================

coef_A12 = np.polyfit(T_pares, A12_lista, 1)
coef_A21 = np.polyfit(T_pares, A21_lista, 1)

def A12_T(T):
    return coef_A12[0] * T + coef_A12[1]

def A21_T(T):
    return coef_A21[0] * T + coef_A21[1]

print("-" * 70)
print("\nRegressões lineares:")
print(f"A12(T) = {coef_A12[0]:.4f} T + {coef_A12[1]:.4f}")
print(f"A21(T) = {coef_A21[0]:.4f} T + {coef_A21[1]:.4f}")

# =========================
# Gráficos A12(T) e A21(T)
# =========================

T_linha = np.linspace(min(T_pares), max(T_pares), 100)

plt.figure(figsize=(8, 6))
plt.scatter(T_pares, A12_lista, s=70, label="A12 ajustado")
plt.plot(T_linha, [A12_T(T) for T in T_linha], linewidth=2, label="Regressão linear")
plt.xlabel("Temperatura (K)")
plt.ylabel("A12")
plt.title("Parâmetro NRTL A12 em função da temperatura")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("A12_vs_T.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(T_pares, A21_lista, s=70, label="A21 ajustado")
plt.plot(T_linha, [A21_T(T) for T in T_linha], linewidth=2, label="Regressão linear")
plt.xlabel("Temperatura (K)")
plt.ylabel("A21")
plt.title("Parâmetro NRTL A21 em função da temperatura")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("A21_vs_T.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# Modelagem da curva binodal
# =========================

def fobj_modelagem(X, T, fator=1.0):
    x_alpha, x_beta = X

    A12 = A12_T(T) * fator
    A21 = A21_T(T) * fator

    eq1 = x_alpha * gamma1(x_alpha, A12, A21, T) - x_beta * gamma1(
        x_beta, A12, A21, T
    )

    eq2 = (1 - x_alpha) * gamma2(x_alpha, A12, A21, T) - (1 - x_beta) * gamma2(
        x_beta, A12, A21, T
    )

    return [eq1, eq2]

def calcular_binodal(T_inicial_K, T_final_K, passo_K=-0.1, fator=1.0):
    T_vals = []
    xa_vals = []
    xb_vals = []

    chute = [x_exp[0], x_exp[-1]]
    T = T_inicial_K

    while T >= T_final_K:
        sol = root(fobj_modelagem, chute, args=(T, fator))

        if sol.success:
            xa, xb = sol.x

            if 0 < xa < xb < 1:
                T_vals.append(T)
                xa_vals.append(xa)
                xb_vals.append(xb)
                chute = [xa, xb]

        T += passo_K

    return np.array(T_vals), np.array(xa_vals), np.array(xb_vals)

T_modelo, xa_modelo, xb_modelo = calcular_binodal(
    T_inicial_K=max(T_pares),
    T_final_K=313.0,
    passo_K=-0.1
)

# =========================
# Gráfico NRTL final
# =========================

plt.figure(figsize=(10, 7))

plt.plot(
    xa_modelo,
    T_modelo - 273.15,
    linewidth=2.5,
    label="Fase α - NRTL"
)

plt.plot(
    xb_modelo,
    T_modelo - 273.15,
    linewidth=2.5,
    label="Fase β - NRTL"
)

plt.scatter(
    x_exp,
    T_exp_C,
    marker="o",
    s=65,
    color="black",
    label="Dados experimentais",
    zorder=5
)

for i, (x, T) in enumerate(zip(x_exp, T_exp_C), start=1):
    plt.annotate(
        str(i),
        (x, T),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9
    )

plt.xlabel("Fração molar de EGME", fontsize=12)
plt.ylabel("Temperatura (°C)", fontsize=12)
plt.title(" NRTL\nSistema EGME + água", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)

T_total = np.concatenate([T_exp_C, T_modelo - 273.15])

plt.ylim(T_total.min() - 2, T_total.max() + 2)
plt.xlim(0.0, max(xb_modelo) + 0.01)

plt.tight_layout()
plt.savefig("modelagem_NRTL_relatorio.png", dpi=300, bbox_inches="tight")
plt.show()