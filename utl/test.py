import matplotlib.pyplot as plt
import pandas as pd

# Cutoff-Datum und Enddatum berechnen (genau 4 Monate)
cutoff_date = pd.to_datetime("2024-02-19")
end_date = cutoff_date + pd.DateOffset(months=4)

# Dummy-Daten für den Plot (12 Monate für Sichtbarkeit)
dates = pd.date_range(start="2023-10-01", end="2025-01-01", freq="D")
values = range(len(dates))

# Erstelle den Plot
plt.figure(figsize=(12, 6))
plt.plot(dates, values, label="Datenverlauf", color="blue")

# ✅ Schraffierten Bereich auf genau 4 Monate begrenzen
plt.axvspan(cutoff_date, end_date, color='gray', alpha=0.2)

# **WICHTIG**: X-Achse begrenzen, damit nicht zu viel angezeigt wird
plt.xlim(pd.to_datetime("2023-10-01"), pd.to_datetime("2025-01-01"))

# Achsentitel und Legende
plt.xlabel("Datum")
plt.ylabel("Wert")
plt.title("Bestandsverlauf mit geschätztem Bereich")
plt.legend()
plt.xticks(rotation=45)

# Zeige das Diagramm
plt.show()