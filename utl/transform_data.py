import pandas as pd
from datetime import datetime
from pathlib import Path
import csv
from itertools import islice
import gc  # Garbage Collector für Speicherfreigabe

import psutil
import os
import time

csv.field_size_limit(100000000000)  # Erhöhe das Limit auf 10 Millionen Zeichen
# Hole den Pfad des Hauptverzeichnisses (2 Ebenen hoch von main.py aus)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Pfad zum "data"-Ordner
DATA_DIR = BASE_DIR / "tests/test_data/NemoDataExport.foc"
# Dateipfade
eingabe_datei = DATA_DIR 
ausgabe_datei = "gefilterte_datei.csv"

# Chunkgröße für speicherschonendes Einlesen
chunk_size = 10

speicher_log = []
# Grenzwert für das Datum (Juni 2024)
grenzdatum = datetime.strptime("2024-01-01", "%Y-%m-%d")

# Spalten, die du behalten möchtest
relevante_spalten = [
                        "PartDesc1",
                        "PartDesc2",
                        "PartDesc3",
                        "PartDesc4", 
                        "ResCapacity(h)",
                        "ProdOrderStorageArea",
                        "ProdOrderCreationDate",
                        "ProdOrderParentOrderOID",
                        #"ProdOrderStatus", 
                        "PartID",
                        "PartGroup",
                        "PartType",
                        "PartTypeDescription",
                        "PartVariant",
                        "PartVariantDesc",
                        "ProdOrderStartDate",
                        "ProdOrderEndDate",
                        "ProdOrderPlanDate",
                        "ProdOrderReqStartDate",
                        "ProdOrderFinishedQty",
                        "ProdOrderTargetQty",
                        "ProdOrderCycleTime",
                        "ProdOrderOptimalLotSize",
                        "ProdOrderOID",
                        "ProdOrderComplDate",
                        "ProdOrderLineStatusDesc",
                        "ProdOrderLineStorageArea",
                        "ProdOrderLineTargetDate",
                        "ProdOrderActCreationDate",
                        "ProdOrderActOperation",
                        "ProdOrderActOID",
                        "ProdOrderActProductionQty",
                        "ProdOrderActActualSetupTime",
                        "ProdOrderActActualUnitTime",
                        "ProdOrderActTargetUnitTime",
                        "ProdOrderActTargetSetupTime",
                        "ProdOrderActEndDate",
                        "ProdOrderActStartDate",
                        "ProdOrderActEndTime",
                        "ProdOrderActStartTime",
                        "ProdOrderActReportedQty",
                        "OrderDocLineCreationDate",
                        "OrderDocLineStorageArea",
                        "OrderDocLineRequestedDate",
                        "OrderDocLineDeliveryDate",
                        #"OrderDocLineOID",
                        #"OrderDocLineOpen",
                        "PurOrderLineCreationDate",
                        "PurOrderDocDate",
                        "PurOrderLineQty",
                        "PurOrderLineDeliveryTime",
                        "PurOrderLineStorageArea",
                        "Company",
                        "ProcessDate"
                        #"PurOrderLineOpen",
                        #"PurOrderSupplierNo"
                        ] # Ersetze mit echten Namen


def filter_kriterien(df, chunk_num):
    """Filtert das DataFrame basierend auf Company und Process Date mit Debugging."""
    try:
        print(f"📌 Chunk {chunk_num}: Vor dem Filtern {len(df)} Zeilen.")
        print(df["Company"].head(5).tolist())  # Zeigt die ersten 5 Werte
        # **Company-Filter**
        df = df[df["Company"].astype(str).str.strip() == "001"]
        print(f"✅ Nach Company-Filter: {len(df)} Zeilen.")
        print(df["Company"].head(5).tolist())  # Zeigt die ersten 5 Werte

        # **Debugging: Zeige einige Werte von Process Date**
        print("📅 Beispielhafte Process Dates VOR der Konvertierung:")
        print(df["ProcessDate"].head(5).tolist())  # Zeigt die ersten 5 Werte

        # **Process Date als Datum konvertieren (ohne Warning)**
        df.loc[:, "ProcessDate"] = pd.to_datetime(df["ProcessDate"].str.strip(), format="%Y-%m-%d", errors="coerce")

        # **Debugging: Zeige ungültige (NaT) Werte**
        nat_count = df["ProcessDate"].isna().sum()
        print(f"⚠️ Anzahl ungültiger (NaT) Datumswerte: {nat_count}")

        # **Nur Zeilen behalten, wo das Datum nach Juni 2024 ist**
        df = df[df["ProcessDate"] > grenzdatum]
        print(f"📊 Nach Process Date-Filter: {len(df)} Zeilen übrig.")

        return df

    except Exception as e:
        print(f"❌ Fehler beim Filtern in Chunk {chunk_num}: {e}")
        return pd.DataFrame()  # Leeres DataFrame zurückgeben

    except Exception as e:
        print(f"❌ Fehler beim Filtern in Chunk {chunk_num}: {e}")
        return pd.DataFrame()  # Leeres DataFrame zurückgeben


def speicherverbrauch():
    """Gibt den aktuellen Speicherverbrauch des Prozesses aus."""
    process = psutil.Process(os.getpid())
    speicher = process.memory_info().rss / (1024 * 1024)  # In MB
    print(f"💾 Aktueller Speicherverbrauch: {speicher:.2f} MB")

# Datei chunkweise einlesen und filtern
chunks = pd.read_csv(
    eingabe_datei,
    #compression="gzip",
    sep=';',
    chunksize=chunk_size,
    dtype=str,  # Alle Daten als String einlesen, um Formatierungsprobleme zu vermeiden
    #usecols=relevante_spalten,  # Nur relevante Spalten laden
    on_bad_lines="skip",  # Pandas 1.3 oder neuer
    engine="python",  # Nutze den flexibleren Parser
    #quotechar='"'  # Falls Zeichen in Anführungszeichen stehen
    low_memory=True 
    
)

# Erste gefilterte Zeilen speichern, danach im Append-Modus weiterschreiben
first_chunk = True
chunk_count = 0  # Zähler für Chunks

for chunk_count,chunk in enumerate(islice(chunks,3000,5000),start=3000):
    try:
        speicherverbrauch()
        start_time = time.time()
        print("start chunk")
        #chunk_count += 1
        print(f"🔄 Verarbeite Chunk {chunk_count} mit {len(chunk)} Zeilen...")
        chunk = chunk.dropna(how="all") 
        # Kopie des Chunks erstellen
        #chunk = chunk.copy()

        # Daten filtern mit zusätzlicher Chunk-Nummer
        gefilterte_zeilen = filter_kriterien(chunk, chunk_count)
        if chunk_count > 2925  and chunk_count < 3100:
            chunk.to_csv(f"debug_chunk_{chunk_count}.csv", index=False)
            print("⚠️ Fehlerhafter Chunk 2928 gespeichert als 'debug_chunk_2928.csv'")

        #break
        # Falls sich der erste Wert drastisch ändert (z. B. "Invoice" statt "001"), speichere den Chunk
        if chunk["Company"].iloc[0] not in ["001", "002", "003", "004", "006", "008"]:  # Setze erlaubte Werte
            print(f"⚠️ WARNUNG! Unerwarteter Wert in 'Company' in Chunk {chunk_count}")
            chunk.to_csv(f"fehlerhafter_chunk_{chunk_count}.csv", index=False)
            continue  # Überspringe diesen fehlerhaften Chunk
        if not gefilterte_zeilen.empty:
            gefilterte_zeilen.to_csv(
                ausgabe_datei,
                mode="w" if first_chunk else "a",
                index=False,
                header=first_chunk  # Header nur beim ersten Schreiben setzen
            )
            print(f"✅ Chunk {chunk_count} gespeichert ({len(gefilterte_zeilen)} Zeilen).")
            first_chunk = False  # Nur setzen, wenn Daten gespeichert wurden
        else:
            print(f"⚠️ Chunk {chunk_count} enthielt keine passenden Zeilen.")

        del chunk
        chunk = None
        del gefilterte_zeilen
        gefilterte_zeilen = None
        gc.collect()        
        
        end_time = time.time()
        speicher = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print([chunk_count, speicher, end_time - start_time])
        speicher_log.append([chunk_count, speicher, end_time - start_time])

    except :
        print(f"⚠️ Fehler in Chunk {chunk_count}:")
        del chunk
        del gefilterte_zeilen
        gc.collect()  # Erzwingt Speicherfreigabe  
        continue  # Fehlerhaften Chunk überspringen


print("🎉 Fertig! Die gefilterte Datei wurde erstellt.")