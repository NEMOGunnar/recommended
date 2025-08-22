#from pa_nemo.nemoqueue.batch_forecasting_logic import *

import numpy as np
import matplotlib.pyplot as plt

# Parameter der EOQ-Kostenfunktion
K = 100   # Bestellfixkosten
D = 5000  # Jahresbedarf
h = 100     # Lagerkosten pro Einheit
N_values = np.linspace(1, 200, 300)  # Wertebereich für die Bestellmenge

# Kostenfunktion
C_values = (K * D / N_values) + (h * N_values / 2)

# Optimale Bestellmenge berechnen (EOQ-Formel)
N_opt = np.sqrt((2 * K * D) / h)
C_min = (K * D / N_opt) + (h * N_opt / 2)

# Plot erstellen
plt.figure(figsize=(10, 5))
plt.plot(N_values, C_values, label=r'$C(N) = \frac{K \cdot D}{N} + \frac{h \cdot N}{2}$', color='black')

# Markierungen hinzufügen
plt.axvline(N_opt, linestyle="dashed", color="gray", label=r'$N_{opt}$')
plt.axhline(C_min, linestyle="dashed", color="gray", label=r'$C_{min}$')

# Achsenbeschriftungen
plt.xlabel("Bestellmenge $N$")
plt.ylabel("Kosten $C$")
plt.title("Optimale Bestellmenge und Kostenverlauf")

# Textlabels für Markierungen
plt.text(N_opt + 2, C_min + 20, r"$C_{min}$", fontsize=12)
plt.text(N_opt - 10, min(C_values) + 50, r"$N_{opt}$", fontsize=12)

# Pfeile für die Richtungen der Kostenentwicklung
plt.annotate("", xy=(N_opt - 50, C_min + 100), xytext=(N_opt - 120, max(C_values) - 200),
             arrowprops=dict(arrowstyle="->", color="black"))
plt.annotate("", xy=(N_opt + 50, C_min + 100), xytext=(N_opt + 120, max(C_values) - 200),
             arrowprops=dict(arrowstyle="->", color="black"))

# Grid und Legende
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Speichern als PNG
plt.savefig("Kostenverlauf.png", dpi=300)

# Anzeigen
plt.show()