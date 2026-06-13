import numpy as np
import matplotlib.pyplot as plt

# Dados da literatura - 298 K
rho = np.array([
    0.9971, 0.9823, 0.9709, 0.9577, 0.9419, 0.9232,
    0.9021, 0.8788, 0.8529, 0.8266, 0.7858
])

x_eth = np.array([
    0.000, 0.033, 0.072, 0.117, 0.171, 0.236,
    0.316, 0.419, 0.552, 0.735, 1.000
])

x_h2o = 1 - x_eth

graus = [2, 3, 4, 5]

for grau in graus:
    coef = np.polyfit(rho, x_h2o, grau)
    poly = np.poly1d(coef)

    x_calc = poly(rho)

    ss_res = np.sum((x_h2o - x_calc)**2)
    ss_tot = np.sum((x_h2o - np.mean(x_h2o))**2)
    r2 = 1 - ss_res/ss_tot

    print(f"\nPolinômio grau {grau}")
    print("Coeficientes:", coef)

    with open("coeficientes.txt", "a", encoding="utf-8") as f:
      f.write(f"\nGrau {grau}\n")
      f.write(f"Coeficientes: {coef}\n")
      f.write(f"R²: {r2}\n")
      print("R²:", r2)

    rho_linha = np.linspace(min(rho), max(rho), 300)

    plt.figure()
    plt.scatter(rho, x_h2o, label="Dados da literatura")
    plt.plot(rho_linha, poly(rho_linha), label=f"Grau {grau}")
    plt.xlabel("Densidade")
    plt.ylabel("x_H2O")
    plt.title(f"Curva de calibração - Grau {grau}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"grau_{grau}.png")
    plt.close()