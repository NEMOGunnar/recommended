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



def plot_production_time_gauss_distribution(df):
    """
    Berechnet die Differenz zwischen 'PROD_ORDER_CREATION_DATE' und 'PROD_ORDER_COMPL_DATE' in Tagen,
    plottet das Histogramm mit einer Balkenbreite von einem Tag und führt ein Fitting einer Gaußverteilung durch.
    
    :param df: Pandas DataFrame mit den Spalten 'PROD_ORDER_CREATION_DATE' und 'PROD_ORDER_COMPL_DATE'
    """
    # Konvertiere die Spalten zu Datumsformat
    df['PROD_ORDER_CREATION_DATE'] = pd.to_datetime(df['PROD_ORDER_CREATION_DATE'], format='%Y-%m-%d')
    df['PROD_ORDER_COMPL_DATE'] = pd.to_datetime(df['PROD_ORDER_COMPL_DATE'], format='%Y-%m-%d')

    # Berechne die Differenz in Tagen
    df['PRODUCTION_DURATION'] = (df['PROD_ORDER_COMPL_DATE'] - df['PROD_ORDER_CREATION_DATE']).dt.days

    # Entferne ungültige oder negative Werte
    df = df[df['PRODUCTION_DURATION'] >= 0]

    # Daten für das Histogramm
    durations = df['PRODUCTION_DURATION'].dropna()

    # Fitten der Gaußverteilung
    mu, std = norm.fit(durations)

    # Erstelle das Histogramm mit einer Balkenbreite von 1 Tag
    plt.figure(figsize=(10, 5))
    counts, bins, patches = plt.hist(durations, bins=np.arange(durations.min(), durations.max() + 1, 1), density=True, alpha=0.6, color='b', edgecolor='black')

    # Erstelle die Wahrscheinlichkeitsverteilung basierend auf der gefitteten Normalverteilung
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Plotte die gefittete Gaußverteilung
    plt.plot(x, p, 'r-', label=f'Normal Fit: μ={mu:.2f}, σ={std:.2f}')

    # Achsenbeschriftungen und Titel
    plt.xlabel('Produktionsdauer (Tage)')
    plt.ylabel('Wahrscheinlichkeit')
    plt.title('Histogramm der Produktionsdauer mit Gauß-Fitting')
    plt.legend()
    
    # Zeige das Diagramm
    plt.show()
    

# Beispielaufruf mit einem vorhandenen DataFrame df
# plot_production_time_distribution(df)


def plot_production_time_chi_square(df):
    """
    Berechnet die Differenz zwischen 'PROD_ORDER_CREATION_DATE' und 'PROD_ORDER_COMPL_DATE' in Tagen,
    plottet das Histogramm mit einer Balkenbreite von einem Tag und führt ein Chi-Quadrat-Fitting durch.
    Außerdem wird die theoretische Chi²-Verteilung geplottet.
    
    :param df: Pandas DataFrame mit den Spalten 'PROD_ORDER_CREATION_DATE' und 'PROD_ORDER_COMPL_DATE'
    """
    # Konvertiere die Spalten zu Datumsformat
    df['PROD_ORDER_CREATION_DATE'] = pd.to_datetime(df['PROD_ORDER_CREATION_DATE'], format='%Y-%m-%d')
    df['PROD_ORDER_COMPL_DATE'] = pd.to_datetime(df['PROD_ORDER_COMPL_DATE'], format='%Y-%m-%d')

    # Berechne die Differenz in Tagen
    df['PRODUCTION_DURATION'] = (df['PROD_ORDER_COMPL_DATE'] - df['PROD_ORDER_CREATION_DATE']).dt.days

    # Entferne ungültige oder negative Werte
    df = df[df['PRODUCTION_DURATION'] >= 0]

    # Daten für das Histogramm
    durations = df['PRODUCTION_DURATION'].dropna()

    # Erstelle das Histogramm mit einer Balkenbreite von 1 Tag (absolute Häufigkeit!)
    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(durations, bins=np.arange(durations.min(), durations.max() + 1, 1), 
                               alpha=0.6, color='b', edgecolor='black', label='Beobachtete Häufigkeiten')

    # Fitting einer Normalverteilung auf die Daten
    mu, sigma = norm.fit(durations)
    x = np.linspace(durations.min(), durations.max(), 100)
    pdf = norm.pdf(x, mu, sigma)

    # Erwartete Häufigkeiten für den Chi-Quadrat-Test **mit exakter Normierung**
    expected_counts = norm.pdf(bins[:-1], mu, sigma)
    expected_counts = expected_counts * (sum(counts) / sum(expected_counts))  # Exakte Anpassung der Summe

    # Chi-Quadrat-Test berechnen (nur wenn alle erwarteten Werte > 0)
    if np.any(expected_counts <= 0):
        print("⚠️ Achtung: Einige erwartete Werte sind 0 oder negativ. Chi²-Test nicht möglich.")
        chi_stat, p_value = None, None
    else:
        chi_stat, p_value = chisquare(f_obs=counts, f_exp=expected_counts)

    # Plotte die angepasste Normalverteilung
    plt.plot(x, pdf * sum(counts), 'r-', label=f'Normalverteilung (μ={mu:.2f}, σ={sigma:.2f})')

    # **Chi²-Verteilung plotten**
    df_chi2 = len(counts) - 1  # Freiheitsgrade = Anzahl der Bins - 1
    chi2_x = np.linspace(0, 2 * chi_stat, 100)  # Bereich für die Chi²-Verteilung
    chi2_pdf = chi2.pdf(chi2_x, df_chi2) * max(counts)  # Skalieren auf Histogramm
    plt.plot(chi2_x, chi2_pdf, 'g--', label=f'Chi²-Verteilung (df={df_chi2})')

    # Anzeige der Chi-Quadrat-Werte (falls berechnet)
    if chi_stat is not None:
        plt.text(bins[-1] - 10, max(counts) * 0.8, f'Chi²: {chi_stat:.2f}\np-Wert: {p_value:.4f}', 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Achsenbeschriftungen und Titel
    plt.xlabel('Produktionsdauer (Tage)')
    plt.ylabel('Anzahl der Produktionsaufträge')
    plt.title('Histogramm der Produktionsdauer mit Chi-Quadrat-Fitting & Chi²-Verteilung')
    plt.legend()
    
    # Zeige das Diagramm
    plt.show()

    # Rückgabe der Chi-Quadrat-Ergebnisse
    return chi_stat, p_value

# Beispielaufruf mit einem vorhandenen DataFrame df
# chi_stat, p_value = plot_production_time_chi_square(df)

def plot_production_time_gmm(df, n_components=3):
    """
    Berechnet die Differenz zwischen 'PROD_ORDER_CREATION_DATE' und 'PROD_ORDER_COMPL_DATE' in Tagen,
    plottet das Histogramm mit einem Gaussian Mixture Model (GMM) Fitting.
    
    :param df: Pandas DataFrame mit den Spalten 'PROD_ORDER_CREATION_DATE' und 'PROD_ORDER_COMPL_DATE'
    :param n_components: Anzahl der GMM-Komponenten (Standard: 3)
    """
    # Konvertiere die Spalten zu Datumsformat
    df['PROD_ORDER_CREATION_DATE'] = pd.to_datetime(df['PROD_ORDER_CREATION_DATE'], format='%Y-%m-%d')
    df['PROD_ORDER_COMPL_DATE'] = pd.to_datetime(df['PROD_ORDER_COMPL_DATE'], format='%Y-%m-%d')

    # Berechne die Differenz in Tagen
    df['PRODUCTION_DURATION'] = (df['PROD_ORDER_COMPL_DATE'] - df['PROD_ORDER_CREATION_DATE']).dt.days

    # Entferne ungültige oder negative Werte
    df = df[df['PRODUCTION_DURATION'] >= 0]

    # Daten für das Histogramm
    durations = df['PRODUCTION_DURATION'].dropna().values.reshape(-1, 1)

    # Histogramm
    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(durations, bins=np.arange(durations.min(), durations.max() + 1, 1),
                               alpha=0.6, color='b', edgecolor='black', density=True, label='Beobachtete Häufigkeiten')

    # **Gaussian Mixture Model (GMM) Fitting**
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(durations)

    # **GMM-Wahrscheinlichkeitsdichte (PDF) berechnen**
    x = np.linspace(durations.min(), durations.max(), 500).reshape(-1, 1)
    gmm_pdf = np.exp(gmm.score_samples(x))  # Wahrscheinlichkeitsdichte für jedes x berechnen

    # **Summierte GMM-Kurve plotten**
    plt.plot(x, gmm_pdf, 'r-', label=f'GMM ({n_components} Komponenten)')

    # Komponenten der GMM einzeln plotten
    for i in range(n_components):
        mean = gmm.means_[i][0]
        std_dev = np.sqrt(gmm.covariances_[i][0, 0])
        weight = gmm.weights_[i]

        # Einzelne Normalverteilungen als gestrichelte Linien zeichnen
        component_pdf = weight * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
        plt.plot(x, component_pdf, linestyle='dashed', label=f'GMM Komponente {i+1}: μ={mean:.2f}, σ={std_dev:.2f}')

    # Achsenbeschriftungen und Titel
    plt.xlabel('Dauer zwischen Anlage und Fertigstellung (PROD_ORDER_COMPL_DATE - PROD_ORDER_CREATION_DATE)')
    plt.ylabel('Dichte')
    plt.title('Anlage und Fertigstellung PPA')
    plt.legend()
    
    # Zeige das Diagramm
    plt.show()

    # Rückgabe des GMM-Modells für weitere Analysen
    return gmm

# Beispielaufruf mit einem vorhandenen DataFrame df
# gmm_model = plot_production_time_gmm(df, n_components=3)
# Beispielaufruf mit einem vorhandenen DataFrame df
# chi_stat, p_value = plot_production_time_chi_square(df)

if __name__ == '__main__':
    columns=[
    'PROD_ORDER_COMPL_DATE',
    'PROD_ORDER_CREATION_DATE'
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

    gmm_model = plot_production_time_gmm(df, n_components=3)