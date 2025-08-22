import datetime
import os

def add_timestamp_to_filename(file_path):
    """
    Fügt dem Dateinamen zwischen Name und Endung einen Zeitstempel hinzu.
    
    Beispiel:
    - Input:  "daten.csv"
    - Output: "daten_20240204_153045.csv"
    
    Format: YYYYMMDD_HHMMSS
    """
    # Aktuelles Datum und Uhrzeit abrufen
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Datei-Verzeichnis, Name und Endung extrahieren
    dir_name, base_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    # Neuen Dateinamen mit Zeitstempel erzeugen
    new_file_name = f"{file_name}_{timestamp}{file_ext}"
    
    # Falls die Datei in einem bestimmten Verzeichnis liegt, diesen Pfad beibehalten
    new_file_path = os.path.join(dir_name, new_file_name) if dir_name else new_file_name

    return new_file_path

if __name__ == "__main__":
    # Beispielnutzung:
    file_path = "mein_report.csv"
    new_path = add_timestamp_to_filename(file_path)
    print("Neue Datei:", new_path)