# ELV água-etanol — Peng-Robinson / abordagem phi-phi
# Versão corrigida para usar componente 1 = ETANOL 
# Estou usando os dados do experimento ELV, depois de acharmos as composições através da calibração
import math
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# BLOCO 1 — ENTRADA DO PROBLEMA
# ===============================
# Componente 1 = ETANOL (para manter x1 e y1 como frações molares de etanol)
Tc1 = 513.9      # K
Pc1 = 61.48      # bar
omega1 = 0.645

# Componente 2 = ÁGUA
Tc2 = 647.1      # K
Pc2 = 220.64     # bar
omega2 = 0.344

P = 1.01325      # bar ~ 1 atm
R = 0.08314      # bar.L/(mol.K)
kij = -0.10  # parâmetro de interação binária PR. Testar sensibilidade depois.

# Dados experimentais do Cenário 1
# T em °C; x1 = x_EtOH líquido; y1 = y_EtOH vapor
# Mantive os pontos negativos para aparecerem no gráfico, mas eles são filtrados nos cálculos de K.

dados_exp = [
    {"T_C": 99.2, "x1": 0.0000, "y1": 0.0000},  # transformei os negativos em zero
    {"T_C": 94.7, "x1": 0.0000, "y1": 0.1613},
    {"T_C": 89.7, "x1": 0.0146, "y1": 0.2716},
    {"T_C": 80.7, "x1": 0.2111, "y1": 0.5281},
    {"T_C": 79.2, "x1": 0.4600, "y1": 0.6397},
    {"T_C": 78.0, "x1": 0.7902, "y1": 0.8009},
]

for p in dados_exp:
    p["T"] = p["T_C"] + 273.15

# Apenas pontos fisicamente válidos para comparação Kexp = y/x
dados_validos = [p for p in dados_exp if 0 < p["x1"] < 1 and 0 < p["y1"] < 1]

# ===============================
# BLOCO 2 — MODELO TERMODINÂMICO (PR)
# ===============================
def pr_pure_parameters(T, Tc, Pc, omega):
    Tr = T / Tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - math.sqrt(Tr)))**2
    a = 0.45724 * (R**2 * Tc**2 / Pc) * alpha
    b = 0.07780 * (R * Tc / Pc)
    return {"Tr": Tr, "kappa": kappa, "alpha": alpha, "a": a, "b": b}


def a12_PR(a1, a2, kij):
    return (a1 * a2) ** 0.5 * (1 - kij)


def amix_bmix_binario(x1, a1, a2, b1, b2, kij):
    x2 = 1 - x1
    a12 = a12_PR(a1, a2, kij)
    amix = x1**2 * a1 + 2 * x1 * x2 * a12 + x2**2 * a2
    bmix = x1 * b1 + x2 * b2
    return {"a12": a12, "amix": amix, "bmix": bmix}


def pr_AB_mistura(x1, a1, a2, b1, b2, kij, T, P):
    mix = amix_bmix_binario(x1, a1, a2, b1, b2, kij)
    A = mix["amix"] * P / (R**2 * T**2)
    B = mix["bmix"] * P / (R * T)
    return {"A": A, "B": B}

# ===============================
# BLOCO 3 — PROPRIEDADES DE FASE (Z)
# ===============================
def raiz_pr_mistura(T, P, x1, kij):
    p1 = pr_pure_parameters(T, Tc1, Pc1, omega1)
    p2 = pr_pure_parameters(T, Tc2, Pc2, omega2)
    a1, b1 = p1["a"], p1["b"]
    a2, b2 = p2["a"], p2["b"]

    AB = pr_AB_mistura(x1, a1, a2, b1, b2, kij, T, P)
    A, B = AB["A"], AB["B"]

    coef = [
        1,
        -(1 - B),
        A - 3 * B**2 - 2 * B,
        -(A * B - B**2 - B**3),
    ]

    roots = np.roots(coef)
    roots_reais = sorted([r.real for r in roots if abs(r.imag) < 1e-6])

    if not roots_reais:
        raise ValueError("Nenhuma raiz real encontrada para Z")

    if len(roots_reais) == 1:
        Z_liquido = Z_vapor = roots_reais[0]
    else:
        Z_liquido = min(roots_reais)
        Z_vapor = max(roots_reais)

    return {"Z_vapor": Z_vapor, "Z_liquido": Z_liquido}

# ===============================
# BLOCO 4 — COEFICIENTES DE FUGACIDADE
# ===============================
def soma_aij_binario(x1, a1, a2, kij, componente):
    x2 = 1 - x1
    a12 = (a1 * a2) ** 0.5 * (1 - kij)
    if componente == 1:
        return x1 * a1 + x2 * a12
    if componente == 2:
        return x1 * a12 + x2 * a2
    raise ValueError("componente deve ser 1 ou 2")


def ln_phi_PR_mistura(Z, A, B, bi, bmix, soma_aij, amix):
    termo1 = (bi / bmix) * (Z - 1)
    termo2 = math.log(abs(Z - B))
    termo3 = A / (2 * math.sqrt(2) * B)
    termo4 = (2 * soma_aij / amix) - (bi / bmix)
    termo5 = math.log((Z + (1 + math.sqrt(2)) * B) / (Z + (1 - math.sqrt(2)) * B))
    return termo1 - termo2 - termo3 * termo4 * termo5


def valores_phi_mistura(T, P, x1, kij):
    p1 = pr_pure_parameters(T, Tc1, Pc1, omega1)
    p2 = pr_pure_parameters(T, Tc2, Pc2, omega2)
    a1, b1 = p1["a"], p1["b"]
    a2, b2 = p2["a"], p2["b"]

    mix = amix_bmix_binario(x1, a1, a2, b1, b2, kij)
    AB = pr_AB_mistura(x1, a1, a2, b1, b2, kij, T, P)
    Z = raiz_pr_mistura(T, P, x1, kij)

    soma1 = soma_aij_binario(x1, a1, a2, kij, 1)
    soma2 = soma_aij_binario(x1, a1, a2, kij, 2)

    ln_phi1_v = ln_phi_PR_mistura(Z["Z_vapor"], AB["A"], AB["B"], b1, mix["bmix"], soma1, mix["amix"])
    ln_phi2_v = ln_phi_PR_mistura(Z["Z_vapor"], AB["A"], AB["B"], b2, mix["bmix"], soma2, mix["amix"])
    ln_phi1_l = ln_phi_PR_mistura(Z["Z_liquido"], AB["A"], AB["B"], b1, mix["bmix"], soma1, mix["amix"])
    ln_phi2_l = ln_phi_PR_mistura(Z["Z_liquido"], AB["A"], AB["B"], b2, mix["bmix"], soma2, mix["amix"])

    return {
        "phi1_v": math.exp(ln_phi1_v), "phi2_v": math.exp(ln_phi2_v),
        "phi1_l": math.exp(ln_phi1_l), "phi2_l": math.exp(ln_phi2_l),
        "Z": Z,
    }

# ===============================
# BLOCO 5 — EQUILÍBRIO
# ===============================
def calcular_K(phi):
    return {"K1": phi["phi1_l"] / phi["phi1_v"], "K2": phi["phi2_l"] / phi["phi2_v"]}


def residuo_T(T, P, x1):
    phi = valores_phi_mistura(T, P, x1, kij)
    K = calcular_K(phi)
    x2 = 1 - x1
    y1 = K["K1"] * x1
    y2 = K["K2"] * x2
    return (y1 + y2) - 1


def resolver_T_bolha(P, x1, T_min=320.0, T_max=390.0):
    f_min = residuo_T(T_min, P, x1)
    f_max = residuo_T(T_max, P, x1)

    if f_min * f_max > 0:
        raise ValueError(f"Intervalo inválido para T em x={x1:.4f}; f_min={f_min:.3g}, f_max={f_max:.3g}")

    for _ in range(80):
        T_mid = 0.5 * (T_min + T_max)
        f_mid = residuo_T(T_mid, P, x1)
        if abs(f_mid) < 1e-7:
            break
        if f_min * f_mid < 0:
            T_max = T_mid
            f_max = f_mid
        else:
            T_min = T_mid
            f_min = f_mid

    phi = valores_phi_mistura(T_mid, P, x1, kij)
    K = calcular_K(phi)
    y1 = K["K1"] * x1
    return T_mid, y1

# ===============================
# BLOCO 6 — AVALIAÇÃO NOS PONTOS EXPERIMENTAIS
# ===============================
print("\n===== Comparação K experimental vs K PR — pontos físicos =====")
for p in dados_validos:
    phi = valores_phi_mistura(p["T"], P, p["x1"], kij)
    K = calcular_K(phi)
    K_exp = p["y1"] / p["x1"]
    erro_rel = abs(K["K1"] - K_exp) / abs(K_exp)
    print(f"T={p['T_C']:5.1f} °C | x={p['x1']:.4f} | y={p['y1']:.4f} | Kexp={K_exp:.4f} | KPR={K['K1']:.4f} | erro={erro_rel:.3f}")


# ===============================
# BLOCO 7 — CURVA TEÓRICA PR
# ===============================
# Intervalo reduzido para evitar regiões extremas onde o PR pode não fechar bem.
x_lista = np.r_[0.000001, np.linspace(0.0001, 0.9999, 300)]

curva_x, curva_y, curva_T = [], [], []
falhas = []
for x1 in x_lista:
    try:
        T_calc, y1_calc = resolver_T_bolha(P, x1)
        if 0 <= y1_calc <= 1:
            curva_x.append(x1)
            curva_y.append(y1_calc)
            curva_T.append(T_calc - 273.15)  # °C
    except Exception as e:
        falhas.append((x1, str(e)))

pontos_modelo = len(curva_x)

#Forçar fechamento nos componentes puros
#curva_x=[0.0] + curva_x + [1.0]
#curva_y=[0.0] + curva_y + [1.0]       
#curva_T=[100.0] + curva_T + [78.37]

#plt.plot(curva_x,curva_T,linewidth=2,label=f"Bolha kij={kij:.2f}")
#plt.plot(curva_y,curva_T,linewidth=2,linestyle="--",label=f"Orvalho kij={kij:.2f}")

print(f"kij={kij:.2f} | pontos calculados = {pontos_modelo} | falhas = {len(falhas)}")

# ===============================
# BLOCO 8 — GRÁFICO T-x-y
# ===============================

x_exp = [p["x1"] for p in dados_exp]
y_exp = [p["y1"] for p in dados_exp]
T_exp_C = [p["T_C"] for p in dados_exp]

plt.figure(figsize=(12, 7))

# Curvas PR
plt.plot(
    curva_x, curva_T,
    linewidth=2.5,
    label="PR - Bolha"
)

plt.plot(
    curva_y, curva_T,
    linewidth=2.5,
    linestyle="--",
    label="PR - Orvalho"
)

# Pontos da literatura
plt.scatter(
    x_exp, T_exp_C,
    s=88,
    marker="o",
    edgecolors="none",
    label="Experimental - líquido"
)

plt.scatter(
    y_exp, T_exp_C,
    s=88,
    marker="s",
    edgecolors="none",
    label="Experimental - vapor"
)

plt.xlabel("Fração molar de etanol", fontsize=16)
plt.ylabel("Temperatura (°C)", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.title(f"Diagrama T-x-y água-etanol — Dados experimentais vs PR (kij = {kij:.2f})", fontsize=18)

plt.xlim(0, 1)
plt.ylim(70,110)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True, fontsize=14)

plt.tight_layout()
plt.savefig(f"diagrama_Txy_PR_experimental_kij_{kij:.2f}.png",
    dpi=600,
    bbox_inches='tight'
)
plt.show()
