from hdbcli import dbapi
import csv
from datetime import date
from timestamp import add_timestamp_to_filename 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2, norm
from sklearn.mixture import GaussianMixture
import matplotlib.dates as mdates

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




def bestandsverlauf_pro_LO_tag(df):
    """
    Erstellt einen vollständigen Bestandsverlauf pro Tag und Lagerort.
    Falls für einen Tag kein Eintrag existiert, wird der letzte bekannte Wert übernommen.
    Falls kein vorheriger Wert existiert, wird 0 gesetzt.

    :param df: Pandas DataFrame mit 'MVMT_STORAGE_AREA', 'MVMT_STORAGE_AREA_ON_HAND', 'MVMT_CREATION_DATE_TIME'
    :return: DataFrame mit vollständigem Bestandsverlauf pro Tag
    """

    # **1. Datumsformat umwandeln**
    df['MVMT_CREATION_DATE_TIME'] = pd.to_datetime(df['MVMT_CREATION_DATE_TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # **2. Neues Feld für das Datum ohne Uhrzeit**
    df['DATE'] = df['MVMT_CREATION_DATE_TIME'].dt.date

    # **3. Daten nach Lagerort und Datum sortieren**
    df_sorted = df.sort_values(by=['MVMT_STORAGE_AREA', 'DATE', 'MVMT_CREATION_DATE_TIME'])

    # **4. Letzten Eintrag pro Tag & Lagerort nehmen**
    df_daily_stock = df_sorted.groupby(['MVMT_STORAGE_AREA', 'DATE']).last().reset_index()

    # **5. Vollständige Datumsreihe erzeugen**
    min_date = df_daily_stock['DATE'].min()
    max_date = df_daily_stock['DATE'].max()
    all_dates = pd.date_range(start=min_date, end=max_date).date

    # **6. Fehlende Daten für jeden Lagerort auffüllen**
    all_storage_areas = df_daily_stock['MVMT_STORAGE_AREA'].unique()
    complete_data = []

    for storage_area in all_storage_areas:
        subset = df_daily_stock[df_daily_stock['MVMT_STORAGE_AREA'] == storage_area].set_index('DATE')
        subset = subset.reindex(all_dates)  # Fehltage auffüllen
        subset['MVMT_STORAGE_AREA'] = storage_area  # Lagerort ergänzen
        subset['MVMT_STORAGE_AREA_ON_HAND'].ffill(inplace=True)  # Fehlende Werte mit vorherigem Bestand füllen
        subset['MVMT_STORAGE_AREA_ON_HAND'].fillna(0, inplace=True)  # Falls es noch keinen vorherigen Wert gab → 0 setzen
        subset.reset_index(inplace=True)  # Index zurück in eine Spalte umwandeln und nicht als Index lassen
        subset.rename(columns={'index': 'DATE'}, inplace=True)  # Datumsspalte korrekt benennen
        complete_data.append(subset)

    # **7. Alles in ein neues DataFrame packen**
    df_complete = pd.concat(complete_data, ignore_index=True)

    # **8. Diagramm für Bestandsverlauf pro Lagerort plotten**
    plt.figure(figsize=(12, 6))
    
    for storage_area in df_complete['MVMT_STORAGE_AREA'].unique():
        subset = df_complete[df_complete['MVMT_STORAGE_AREA'] == storage_area]
        plt.plot(subset['DATE'], subset['MVMT_STORAGE_AREA_ON_HAND'], marker='o', linestyle='-', label=f'Lager {storage_area}')

    # **9. Diagramm-Formatierung**
    plt.xlabel('Datum')
    plt.ylabel('Bestand')
    plt.title('Bestandsverlauf pro Lagerort mit vollständiger Zeitreihe')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # **10. Diagramm anzeigen**
    plt.show()

    return df_complete

def gesamter_bestandsverlauf(df):
    """
    Erstellt einen vollständigen Bestandsverlauf pro Tag basierend auf 'MVMT_ON_HAND'.
    Falls für einen Tag kein Eintrag existiert, wird der letzte bekannte Wert übernommen.
    Falls kein vorheriger Wert existiert, wird 0 gesetzt.

    :param df: Pandas DataFrame mit 'MVMT_ON_HAND' und 'MVMT_CREATION_DATE_TIME'
    :return: DataFrame mit vollständigem Bestandsverlauf pro Tag
    """

    # **1. Datumsformat umwandeln**
    df['MVMT_CREATION_DATE_TIME'] = pd.to_datetime(df['MVMT_CREATION_DATE_TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # **2. Neues Feld für das Datum ohne Uhrzeit**
    df['DATE'] = df['MVMT_CREATION_DATE_TIME'].dt.date

    # **3. Daten nach Datum und Zeit sortieren**
    df_sorted = df.sort_values(by=['DATE', 'MVMT_CREATION_DATE_TIME'])

    # **4. Letzten Bestand pro Tag bestimmen (aus 'MVMT_ON_HAND')**
    df_daily_stock = df_sorted.groupby('DATE')['MVMT_ON_HAND'].last().reset_index()

    # **5. Vollständige Datumsreihe erzeugen**
    min_date = df_daily_stock['DATE'].min()
    max_date = df_daily_stock['DATE'].max()
    all_dates = pd.date_range(start=min_date, end=max_date).date

    # **6. Fehlende Daten auffüllen**
    df_daily_stock = df_daily_stock.set_index('DATE').reindex(all_dates)  # Fehlende Tage hinzufügen
    df_daily_stock['MVMT_ON_HAND'].ffill(inplace=True)  # Fehlende Werte mit vorherigem Bestand füllen
    df_daily_stock['MVMT_ON_HAND'].fillna(0, inplace=True)  # Falls es noch keinen vorherigen Wert gab → 0 setzen
    df_daily_stock.reset_index(inplace=True)  # Index zurück in eine Spalte umwandeln
    df_daily_stock.rename(columns={'index': 'DATE'}, inplace=True)  # Spalte korrekt benennen

    # **7. Diagramm für Gesamtbestandsverlauf plotten**
    plt.figure(figsize=(12, 6))
    plt.plot(df_daily_stock['DATE'], df_daily_stock['MVMT_ON_HAND'], marker='o', linestyle='-', color='b', label='Gesamtbestand')

    # **8. Diagramm-Formatierung**
    plt.xlabel('Datum')
    plt.ylabel('Gesamtbestand')
    plt.title('Gesamtbestandsverlauf basierend auf MVMT_ON_HAND')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # **9. Diagramm anzeigen**
    #plt.show()

    return df_daily_stock

# **Beispielaufruf**
# df_gesamtbestand = gesamter_bestandsverlauf(df)


def bestandsverlauf_moving_average(df, window=14):
    """
    Erstellt einen Bestandsverlauf mit gleitendem Mittelwert über eine bestimmte Anzahl an Tagen.

    :param df: Pandas DataFrame mit 'DATE' und 'MVMT_ON_HAND'
    :param window: Fenstergröße für den gleitenden Mittelwert (Standard: 14 Tage)
    :return: DataFrame mit gleitendem Mittelwert
    """

    # **1. Sicherstellen, dass DATE eine Zeitreihe ist**
    df['DATE'] = pd.to_datetime(df['DATE'])

    # **2. Gleitenden Mittelwert berechnen (14-Tage-Schnitt)**
    df['MVMT_ON_HAND_MA'] = df['MVMT_ON_HAND'].rolling(window=window, min_periods=1).mean()

    # **3. Diagramm für Bestandsverlauf mit gleitendem Mittelwert plotten**
    plt.figure(figsize=(12, 6))

    # Original-Bestand plotten
    plt.plot(df['DATE'], df['MVMT_ON_HAND'], marker='o', linestyle='-', color='b', alpha=0.4, label='Täglicher Bestand')

    # Gleitenden Mittelwert plotten
    plt.plot(df['DATE'], df['MVMT_ON_HAND_MA'], marker='', linestyle='-', color='r', linewidth=2, label=f'{window}-Tage Moving Average')

    # **4. Diagramm-Formatierung**
    plt.xlabel('Datum')
    plt.ylabel('Bestand')
    plt.title(f'Bestandsverlauf mit {window}-Tage Moving Average')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # **5. Diagramm anzeigen**
    #plt.show()

    return df


def process_inventory_trends_with_moving_average(df, initial_stock=None, window=14):
    """
    Berechnet den Bestandsverlauf basierend auf Produktion, Verkauf und Einkauf.
    Zusätzlich wird ein gleitender Mittelwert (Moving Average) für die reale und geschätzte Bestandskurve berechnet.
    
    :param df: Pandas DataFrame mit Bestandsdaten
    :param initial_stock: Optionaler Startbestand, falls keiner aus df['MVMT_ON_HAND'] genutzt werden soll.
    :param window: Fenstergröße für den Moving Average (Standard: 14 Tage)
    :return: DataFrame mit geschätztem Bestand & realem Bestandsverlauf
    """

    # ✅ Sicherstellen, dass alle Datumsfelder als datetime erkannt werden
    date_fields = ['PROD_ORDER_END_DATE', 'ORDER_DOC_LINE_REQUESTED_DATE', 'PUR_ORDER_LINE_REQUESTED_DATE']
    
    for field in date_fields:
        if field in df.columns:
            df[field] = pd.to_datetime(df[field], errors='coerce')

    # ✅ Produktion extrahieren (negative Menge, da Verbrauch)
    prod_df = df[['PROD_ORDER_END_DATE', 'PROD_ORDER_TARGET_QTY']].dropna()
    prod_df = prod_df.rename(columns={'PROD_ORDER_END_DATE': 'DATE', 'PROD_ORDER_TARGET_QTY': 'QTY'})
    prod_df['QTY'] *= -1  

    # ✅ Verkauf extrahieren (negative Menge, da Lagerabgang)
    order_df = df[['ORDER_DOC_LINE_REQUESTED_DATE', 'ORDER_DOC_LINE_QTY']].dropna()
    order_df = order_df.rename(columns={'ORDER_DOC_LINE_REQUESTED_DATE': 'DATE', 'ORDER_DOC_LINE_QTY': 'QTY'})
    order_df['QTY'] *= -1  

    # ✅ Einkauf extrahieren (positive Menge, da Lagerzugang)
    pur_df = df[['PUR_ORDER_LINE_REQUESTED_DATE', 'PUR_ORDER_LINE_QTY']].dropna()
    pur_df = pur_df.rename(columns={'PUR_ORDER_LINE_REQUESTED_DATE': 'DATE', 'PUR_ORDER_LINE_QTY': 'QTY'})
    pur_df['QTY'] *= 1  

    # ✅ Alle Bewegungen zusammenführen
    all_data = pd.concat([prod_df, order_df, pur_df])

    # ✅ **Doppelte Datumswerte aggregieren**
    daily_inventory = all_data.groupby('DATE', as_index=False)['QTY'].sum()

    # ✅ **Lückenlose Zeitreihe erstellen (fehlende Tage auffüllen)**
    all_dates = pd.date_range(start=daily_inventory['DATE'].min(), end=daily_inventory['DATE'].max(), freq='D')
    daily_inventory = daily_inventory.set_index('DATE').reindex(all_dates, fill_value=np.nan)

    # ✅ **Fehlende Werte mit 0 füllen**
    daily_inventory['QTY'] = daily_inventory['QTY'].fillna(0)

    # ✅ **Anfangsbestand bestimmen**
    if initial_stock is None:
        initial_stock = df['MVMT_ON_HAND'].dropna().iloc[0]  # Falls kein Wert gegeben, nehme ersten verfügbaren Bestand

    # ✅ **Kumulierte Bestandsberechnung**
    daily_inventory['CUM_QTY'] = initial_stock + daily_inventory['QTY'].cumsum()

    # ✅ **Moving Average für geschätzte Bestandskurve**
    daily_inventory['CUM_QTY_MA'] = daily_inventory['CUM_QTY'].rolling(window=window, min_periods=1).mean()

    # ✅ **Realen Bestand mit Moving Average berechnen**
    real_inventory = df[['MVMT_CREATION_DATE_TIME', 'MVMT_ON_HAND']].dropna()
    real_inventory = real_inventory.rename(columns={'MVMT_CREATION_DATE_TIME': 'DATE', 'MVMT_ON_HAND': 'REAL_QTY'})
    real_inventory = real_inventory.groupby('DATE', as_index=False).last().set_index('DATE')
    real_inventory = real_inventory.reindex(all_dates, method='ffill')  # Fehlende Werte mit letztem Wert füllen

    # ✅ **Moving Average für realen Bestand**
    real_inventory['REAL_QTY_MA'] = real_inventory['REAL_QTY'].rolling(window=window, min_periods=1).mean()

    # ✅ **Bestandskurven kombinieren**
    final_inventory = daily_inventory.merge(real_inventory, left_index=True, right_index=True, how='outer')

# ✅ **Werte für Titel generieren**
    title_fields = ["PART_I_D", "PART_DESC1", "PART_DESC2", "PART_DESC3", "PART_DESC4"]
    title_values = [str(df[field].dropna().iloc[0]) if field in df.columns and not df[field].dropna().empty else "N/A" for field in title_fields]
    title_text = " - ".join(title_values)

    # ✅ **Plotten der Bestandskurven**
    plt.figure(figsize=(12, 6))

    # Geschätzter Bestand
    plt.plot(final_inventory.index, final_inventory['CUM_QTY'], linestyle='-', color='b', label="Geschätzter Bestand")

    # Gleitender Mittelwert für geschätzten Bestand
    plt.plot(final_inventory.index, final_inventory['CUM_QTY_MA'], linestyle='-', color='orange', linewidth=2, label=f'{window}-Tage MA Geschätzter Bestand')

    # Reale Bestandswerte
    plt.plot(final_inventory.index, final_inventory['REAL_QTY'], linestyle='--', color='red', alpha=0.5, label="Realer Bestand")

    # Gleitender Mittelwert des realen Bestands
    plt.plot(final_inventory.index, final_inventory['REAL_QTY_MA'], linestyle='-', color='green', linewidth=2, label=f'{window}-Tage MA Realer Bestand')

    # **Diagramm Formatierung**
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.xlabel("Datum")
    plt.ylabel("Bestand")
    plt.title(f"Bestandsverlauf: {title_text}")
    plt.legend()
    plt.grid()
    # ✅ **Dateiname für die Grafik setzen**
    part_id = str(df['PART_I_D'].dropna().iloc[0]) if 'PART_I_D' in df.columns and not df['PART_I_D'].dropna().empty else "UNKNOWN"
    file_name = f"Bestandsverlauf_{part_id}.png"

    # ✅ **Grafik speichern**
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"🔹 Grafik gespeichert als: {file_name}")
    

    return final_inventory


# ✅ **Beispielaufruf**
# final_inventory_df = process_inventory_trends_with_moving_average(df, initial_stock=50000, window=14)


# ✅ Beispielaufruf mit df
# daily_inventory_df = process_inventory_trends(df)

# **Beispielaufruf**
# df_moving_avg = bestandsverlauf_moving_average(df_gesamtbestand, window=14)

if __name__ == '__main__':
    columns=[
    'MVMT_STORAGE_AREA',
    'MVMT_STORAGE_AREA_ON_HAND',
    'MVMT_CREATION_DATE_TIME',
    'MVMT_ON_HAND',
    'PROD_ORDER_END_DATE',
    'PROD_ORDER_TARGET_QTY',
    'ORDER_DOC_LINE_QTY', 
    'ORDER_DOC_LINE_REQUESTED_DATE',
    'PUR_ORDER_LINE_QTY',
    'PUR_ORDER_LINE_REQUESTED_DATE',
    "PART_DESC1", 
    "PART_DESC2",
    "PART_DESC3",
    "PART_DESC4",
    "PART_I_D"
    ]
    part_list = [
        '31612046',
        '031800112',
        '0240121',
        '044000131',
        '0630015',
        '080205401',
        '0239205',
        '0241030',
        '031800010',
        '0802009',
        '0803026',
        '080206140',
        '0631678',
        '0239204',
        '0803090',
        '0801006',
        '202074070',
        '080206120',
        '031801212',
        '024599490',
        '0630256'
        ]
    column_list = ", ".join(f'"{col}"' for col in columns)
    grenzdatum_start = date(2023, 1, 1)  # Python DATE-Objekt
    grenzdatum_ende = date(2024, 12, 1)

    for part in part_list:

        query = f"""
            SELECT {column_list} 
            FROM "NEMO"."pa_export_apra" 
            WHERE "COMPANY" = ?
            AND TO_DATE("PROCESS_DATE", 'YYYY-MM-DD') > ?
            AND TO_DATE("PROCESS_DATE", 'YYYY-MM-DD') < ?
            AND (PROCESS = 'Make'
            OR PROCESS = 'Production'
            OR PROCESS = 'Source'
            OR PROCESS = 'Deliver'
            )
            AND PART_I_D = '{part}'
        ;
        """

        params = ("001", grenzdatum_start.strftime("%Y-%m-%d"), grenzdatum_ende.strftime("%Y-%m-%d"))  # Übergabe als String
        results = execute_query(conn, query,params=params)
        
        # Erstellen eines DataFrame aus den Ergebnissen
        df = pd.DataFrame(results, columns=columns)
        # df_bestand = bestandsverlauf_pro_LO_tag(df)
        try:
            df_gesamtbestand = gesamter_bestandsverlauf(df)
            df_moving_avg = bestandsverlauf_moving_average(df_gesamtbestand, window=14)
            final_inventory_df = process_inventory_trends_with_moving_average(df, initial_stock=None, window=34)
        except:
            print('Fehler in: ' + part)   
    conn.close()
    plt.show()
    #'PROD_ORDER_END_DATE', 'PROD_ORDER_TARGET_QTY',
    # 'ORDER_DOC_LINE_QTY', 'ORDER_DOC_LINE_REQUESTED_DATE',
    # 'PUR_ORDER_LINE_QTY', 'PUR_ORDER_LINE_REQUESTED_DATE',