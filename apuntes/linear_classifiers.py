"""
Example: Let h be the linear classifier defined by 
θ = [ -1
     1.5
    ] 
, θ0=3.

X1 and X2 are specify in the data set 

h(x; θ, θ₀) = sign(θᵀ x + θ₀) =
    +1   if  θᵀ x + θ₀ > 0
    -1   otherwise
"""


with open ("./linear_classifier_no_hx.txt") as f:
    for x in f:
        print(x)

import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar dataset (sin h(x), con encabezado)
data = np.loadtxt("linear_classifier_no_hx.txt", skiprows=1)

# 2. Separar columnas
x1 = data[:, 0]
x2 = data[:, 1]

# 3. Parámetros del clasificador
theta = np.array([-1, 1.5])
theta0 = 3

# 4. h(x)
scores = theta[0]*x1 + theta[1]*x2 + theta0
labels = np.sign(scores)

# 5. Gráfica
plt.figure(figsize=(7,7))

plt.scatter(x1[labels>0], x2[labels>0], c="blue", label="Positive (+1)", s=12)
plt.scatter(x1[labels<0], x2[labels<0], c="red", label="Negative (-1)", s=12)

# 6. Línea de decisión
x_vals = np.linspace(min(x1), max(x1), 300)
y_vals = -(theta[0]*x_vals + theta0) / theta[1]

plt.plot(x_vals, y_vals, c="black", linewidth=1.5, label="Decision boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Linear Classifier h(x) = sign(theta·x + theta0)")
plt.legend()
plt.grid(True)
plt.show()
