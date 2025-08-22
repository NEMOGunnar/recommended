import pandas as pd
import glob
from datetime import datetime

# 🛠 Konfiguration
input_pattern = "split_*.csv"  # Alle geteilten CSV-Dateien
output_file = "filtered_combined_006.csv"  # Finale gefilterte Datei
chunk_size = 50000  # Anzahl der Zeilen pro Chunk
grenzdatum_start = datetime.strptime("2022-01-01", "%Y-%m-%d")
grenzdatum_end = datetime.strptime("2023-01-01", "%Y-%m-%d")
# **Relevante Spalten**
relevante_spalten = [
    "PartDesc1", "PartDesc2", "PartDesc3", "PartDesc4",
    "ResCapacity(h)", "ProdOrderStorageArea", "ProdOrderCreationDate",
    "ProdOrderParentOrderOID", "PartID", "PartGroup", "PartType",
    "PartTypeDescription", "PartVariant", "PartVariantDesc",
    "ProdOrderStartDate", "ProdOrderEndDate", "ProdOrderPlanDate",
    "ProdOrderReqStartDate", "ProdOrderFinishedQty", "ProdOrderTargetQty",
    "ProdOrderCycleTime", "ProdOrderOptimalLotSize", "ProdOrderOID",
    "ProdOrderComplDate", "ProdOrderLineStatusDesc",
    "ProdOrderLineStorageArea", "ProdOrderLineTargetDate",
    "ProdOrderActCreationDate", "ProdOrderActOperation", "ProdOrderActOID",
    "ProdOrderActProductionQty", "ProdOrderActActualSetupTime",
    "ProdOrderActActualUnitTime", "ProdOrderActTargetUnitTime",
    "ProdOrderActTargetSetupTime", "ProdOrderActEndDate",
    "ProdOrderActStartDate", "ProdOrderActEndTime", "ProdOrderActStartTime",
    "ProdOrderActReportedQty", "OrderDocLineCreationDate",
    "OrderDocLineStorageArea", "OrderDocLineRequestedDate",
    "OrderDocLineDeliveryDate", "PurOrderLineCreationDate",
    "PurOrderDocDate", "PurOrderLineQty", "PurOrderLineDeliveryTime",
    "PurOrderLineStorageArea", "Company", "ProcessDate"
]

# **Alle CSV-Dateien suchen**
csv_files = sorted(glob.glob(input_pattern))
if not csv_files:
    print("❌ Keine Dateien gefunden!")
    exit()

print(f"📂 {len(csv_files)} Dateien gefunden. Starte Verarbeitung...")

first_file = True  # Steuert das Hinzufügen des Headers

# **Alle Dateien einzeln einlesen und verarbeiten**
for file in csv_files:
    print(f"🔄 Verarbeite Datei: {file}")
    
    try:
        for chunk in pd.read_csv(file, usecols=relevante_spalten, chunksize=chunk_size, dtype=str, sep=";"):
            # **Daten filtern**
            chunk = chunk[chunk["Company"].str.strip() == "001"]  # Nur Company "001"
            
            # **ProcessDate in datetime konvertieren**
            chunk["ProcessDate"] = pd.to_datetime(chunk["ProcessDate"], format="%Y-%m-%d", errors="coerce")
            
            # **Nach gültigem Datum filtern**
            chunk = chunk[(chunk["ProcessDate"] > grenzdatum_start) & (chunk["ProcessDate"] < grenzdatum_end)]
            
            if chunk.empty:
                print(f"⚠️ Datei {file}: Keine gültigen Daten nach Filterung.")
                continue
            
            # **Gefilterte Daten speichern**
            chunk.to_csv(output_file, mode="w" if first_file else "a", index=False, header=first_file)
            first_file = False  # Nur beim ersten Durchlauf Header setzen
            
            print(f"✅ {len(chunk)} Zeilen gespeichert aus Datei {file}")
    
    except Exception as e:
        print(f"❌ Fehler beim Verarbeiten von {file}: {e}")

print("🎉 Verarbeitung abgeschlossen! Die gefilterte Datei wurde erstellt.")