from hdbcli import dbapi
import csv
from datetime import date
from timestamp import add_timestamp_to_filename 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2, norm
from sklearn.mixture import GaussianMixture

conn = dbapi.connect(
    address="10.0.0.80",
    port=30015,
    user="CUSTOMER_ADMIN",
    password="mIBFtYdgmE4nh0VaNkJS",
)

def execute_query(connection, query, params=None):
    cursor = connection.cursor()
    cursor.execute(query, params or ())
    results = cursor.fetchall()
    #column_names = [desc[0] for desc in cursor.description]
    #print("Spaltennamen:", column_names)
    cursor.close()
    return results



def plot_production_qty_gmm(df, n_components=3):
    """
    Berechnet die Differenz zwischen 'PROD_ORDER_TARGET_QTY' und 'PROD_ORDER_FINISHED_QTY',
    plottet das Histogramm mit einem Gaussian Mixture Model (GMM) Fitting,
    wobei jeder Balken genau 1 Tag repräsentiert.

    :param df: Pandas DataFrame mit den Spalten 'PROD_ORDER_TARGET_QTY' und 'PROD_ORDER_FINISHED_QTY'
    :param n_components: Anzahl der GMM-Komponenten (Standard: 3)
    """
    # Konvertiere zu numerischen Werten (Fehlerhafte Einträge zu NaN)
    df['PROD_ORDER_TARGET_QTY'] = pd.to_numeric(df['PROD_ORDER_TARGET_QTY'], errors='coerce')
    df['PROD_ORDER_FINISHED_QTY'] = pd.to_numeric(df['PROD_ORDER_FINISHED_QTY'], errors='coerce')

    # Berechne die Differenz zwischen Ziel- und Ist-Produktion
    df['PRODUCTION_DIFF'] = df['PROD_ORDER_FINISHED_QTY'] - df['PROD_ORDER_TARGET_QTY']

    # Entferne NaN-Werte
    df = df.dropna(subset=['PRODUCTION_DIFF'])

    # Werte für das Histogramm extrahieren
    production_diffs = df['PRODUCTION_DIFF'].values

    # Automatische Berechnung der Bin-Anzahl
    min_diff = int(np.floor(production_diffs.min()))
    print(min_diff)
    max_diff = int(np.ceil(production_diffs.max()))
    print(max_diff)
    bins = np.arange(min_diff, max_diff + 1, 1)  # 1 Tag pro Bin
    print(bins)

    # Histogramm
    plt.figure(figsize=(12, 6))
    counts, bins_edges, _ = plt.hist(production_diffs, bins=bins, alpha=0.6, color='b', edgecolor='black', density=True, label='Beobachtete Häufigkeiten')

    # **Gaussian Mixture Model (GMM) Fitting**
    production_diffs = production_diffs.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(production_diffs)

    # **GMM-Wahrscheinlichkeitsdichte (PDF) berechnen**
    x = np.linspace(min_diff, max_diff, 500).reshape(-1, 1)
    gmm_pdf = np.exp(gmm.score_samples(x))  # Wahrscheinlichkeitsdichte für jedes x berechnen

    # **Summierte GMM-Kurve plotten**
    plt.plot(x, gmm_pdf, 'r-', label=f'GMM ({n_components} Komponenten)')

    # Einzelne GMM-Komponenten plotten
    for i in range(n_components):
        mean = gmm.means_[i][0]
        std_dev = np.sqrt(gmm.covariances_[i][0, 0])
        weight = gmm.weights_[i]

        # Einzelne Normalverteilungen als gestrichelte Linien zeichnen
        component_pdf = weight * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
        plt.plot(x, component_pdf, linestyle='dashed', label=f'GMM Komponente {i+1}: μ={mean:.2f}, σ={std_dev:.2f}')

    # Achsenbeschriftungen und Titel
    plt.xlabel('Produktionsdifferenz (PROD_ORDER_FINISHED_QTY - PROD_ORDER_TARGET_QTY) in Tagen')
    plt.ylabel('Dichte')
    plt.title('Differenz Produktionsmengen Ist-Soll')
    plt.xticks(bins, rotation=45)  # Exakte Tageswerte anzeigen
    plt.legend()
    
    # Zeige das Diagramm
    plt.show()

    # Rückgabe des GMM-Modells für weitere Analysen
    return gmm

# Beispielaufruf mit einem vorhandenen DataFrame df
# gmm_model = plot_production_qty_gmm_daily(df, n_components=3)

# Beispielaufruf mit einem vorhandenen DataFrame df
# gmm_model = plot_production_qty_gmm(df, n_components=3)

if __name__ == '__main__':
    columns=[
    'PROD_ORDER_TARGET_QTY',
    'PROD_ORDER_FINISHED_QTY',
    ]

    column_list = ", ".join(f'"{col}"' for col in columns)
    grenzdatum_start = date(2023, 1, 1)  # Python DATE-Objekt
    grenzdatum_ende = date(2024, 12, 1)

    query = f"""
        SELECT {column_list} 
        FROM "NEMO"."pa_export_apra" 
        WHERE "COMPANY" = ?
        AND TO_DATE("PROCESS_DATE", 'YYYY-MM-DD') > ?
        AND TO_DATE("PROCESS_DATE", 'YYYY-MM-DD') < ?
        AND PROCESS = 'Production'
        AND PART_I_D = '0630015'
    ;
    """

    params = ("001", grenzdatum_start.strftime("%Y-%m-%d"), grenzdatum_ende.strftime("%Y-%m-%d"))  # Übergabe als String
    results = execute_query(conn, query,params=params)

    # Erstellen eines DataFrame aus den Ergebnissen
    df = pd.DataFrame(results, columns=columns)

    gmm_model = plot_production_qty_gmm(df, n_components=3)