import numpy as np
import matplotlib.pyplot as plt

# Parameter
a = 1
b = 0.5
c = 2

# Definitionsbereich: x > 0 (wegen ln(x))
x = np.linspace(0.1, 10, 500)
f_x = np.log(x)**c + x * (a + b * x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, f_x, label=r'$f(x) = \ln(x)^c + x(a + bx)$')
plt.title('Plot der Funktion')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()