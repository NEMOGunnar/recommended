import csv
from pathlib import Path
# Hole den Pfad des Hauptverzeichnisses (2 Ebenen hoch von main.py aus)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Pfad zum "data"-Ordner
DATA_DIR = BASE_DIR / "tests/test_data/NemoDataExport.foc"
# Dateipfade
eingabe_datei = DATA_DIR 
# Konfigurierbare Parameter
input_file = eingabe_datei  # Pfad zur großen CSV
output_prefix = "split_"  # Präfix für die Ausgabedateien
rows_per_file = 100  # Anzahl der Zeilen pro Datei
import sys
csv.field_size_limit(sys.maxsize)  # Erhöhe das Limit auf 10 Millionen Zeichen

rows_per_file = 100  # Kleinere Chunks für weniger RAM-Nutzung

def split_csv(input_file, output_prefix, rows_per_file):
    """Teilt eine große CSV-Datei in kleinere Dateien auf."""
    file_count = 1
    row_count = 0

    with open(input_file, "r", encoding="utf-8", errors="replace") as infile:
        header = infile.readline().strip()  # Lese nur die erste Zeile als Header
        
        output_file = f"{output_prefix}{file_count}.csv"
        outfile = open(output_file, "w", encoding="utf-8", errors="replace")
        outfile.write(header + "\n")

        for line in infile:
            if row_count >= rows_per_file:
                outfile.close()
                file_count += 1
                output_file = f"{output_prefix}{file_count}.csv"
                outfile = open(output_file, "w", encoding="utf-8", errors="replace")
                outfile.write(header + "\n")
                row_count = 0

            outfile.write(line)
            row_count += 1

        outfile.close()

    print(f"✅ Datei erfolgreich in {file_count} kleinere Dateien gesplittet!")

split_csv(input_file, output_prefix, rows_per_file)