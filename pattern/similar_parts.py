import numpy as np
import pandas as pd
import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
import utl
from sentence_transformers import SentenceTransformer

class SimilarParts:

    def __init__(self, load_last_data = False, filepath=""):
        if load_last_data == False:
            self.read_gzip(filepath)
            self.min_threshold = 0.1
    
    def read_gzip(self, file_path):
        df = pd.read_csv(file_path, compression="gzip",sep= ";",encoding="utf-8",  nrows=100 ,
                 usecols=[
                        "PartDesc1",
                        "PartDesc2",
                        "PartDesc3",
                        "PartDesc4", 
                        "PartID",
                        "PartGroup",
                        "PartType",
                        "PartTypeDescription",
                        "PartVariant",
                        "PartVariantDesc"
                        ]).dropna(subset=[
                        "PartID" 
                                          ]).drop_duplicates(subset=["PartID"])

    
        # Speichern der ersten 100 Zeilen in eine CSV-Datei
        output_file = "tests/test_data/output/Parts.csv"
        time_stamp_file = utl.add_timestamp_to_filename(output_file)
        # df.to_csv(time_stamp_file, index=False, sep=";")
        self.data = df
        self.part_ids = list(self.data["PartID"])  # Liste der PartIDs für den Index-Zugriff
        print(f"gespeichert in: {time_stamp_file}")  

    def init_similar_matrix(self):
        self.data["PartDescComb" ] = self.data["PartDesc1" ] + " " + self.data["PartDesc2" ] + " " + self.data["PartDesc3" ] + " " + self.data["PartDesc4" ]
        # 🔹 Lade das Deep Learning Modell für Text-Embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # 🔹 Berechne die Embeddings für alle Artikel
        text_embeddings = model.encode(self.data["PartDescComb" ].tolist(), show_progress_bar=True)
        # 🔹 Ähnlichkeitsmatrix mit Cosine Similarity berechnen
        self.similarity_matrix = cosine_similarity(text_embeddings)

        # 🔹 Zeigt die Ähnlichkeitsmatrix
        # print("Cosine Similarity Matrix:\n", self.similarity_matrix)

    # 🔹 Funktion: Top 3 ähnlichste Artikel zu einem gegebenen Artikel finden
    def get_top_similar_articles(self,article_index,top_n=3):
        similarity_matrix = self.similarity_matrix
        df = self.data
        similar_indices = np.argsort(-similarity_matrix[article_index])[1:top_n+1]  # Ignoriere sich selbst
        similar_articles = df.iloc[similar_indices]
        artikel = df.iloc[article_index]
        print(artikel["PartID"] + ': ' + artikel["PartDesc1"] + ' ' + artikel["PartDesc2"] + ' ' + artikel["PartDesc3"] + ' ' + artikel["PartDesc4"] )
        return similar_articles[["PartID", "PartDesc1", "PartDesc2", "PartDesc3", "PartDesc4"]]
    
    def get_similar_parts(self, selected_parts):
        """
        Berechnet die Top 10 ähnlichsten Artikel für eine gegebene Liste von Artikeln.

        Args:
            selected_parts (list): Liste der gewählten PartIDs.

        Returns:
            dict: JSON-kompatible Antwort mit gewählten und ähnlichen Artikeln.
        """
        print(f"📌 Eingehende Anfrage: {selected_parts}")

        # 🔹 Konvertiere PartIDs zu Index-Werten
        selected_indices = [self.part_ids.index(pid) for pid in selected_parts if pid in self.part_ids]

        if not selected_indices:
            return {"error": "No valid parts selected"}

        # 🔹 Berechnung der Ähnlichkeitswerte (Summiere die Scores)
        similar_scores = np.sum(self.similarity_matrix[selected_indices], axis=0)

        # 🔹 Sortiere Artikel nach höchster Ähnlichkeit
        sorted_similar_parts = sorted(zip(self.part_ids, similar_scores), key=lambda x: -x[1])[:10]

        # 🔹 Erstelle die Rückgabeantwort
        return {
            "selected_parts": [
                    {
                        "part_id": row["PartID"],
                        "name": row["PartID"],  # Falls ein Name existiert, kann dieser hier ergänzt werden
                        "description": self.get_description(row)
                    }
                    for _, row in self.data[self.data["PartID"].isin(selected_parts)].iterrows()
                ],
            "similar_parts": [
                {
                    "part_id": pid,
                    "score": float(score),
                    "name": self.data.loc[self.data["PartID"] == pid, "PartID"].values[0],
                    "description": self.data.loc[self.data["PartID"] == pid, "PartDesc1"].values[0] 
                    + ' ' + self.data.loc[self.data["PartID"] == pid, "PartDesc2"].values[0]
                    + ' ' + self.data.loc[self.data["PartID"] == pid, "PartDesc3"].values[0]
                    + ' ' + self.data.loc[self.data["PartID"] == pid, "PartDesc4"].values[0]
                    ,
                }
                for pid, score in sorted_similar_parts if pid not in selected_parts  # Gewählte Artikel ausschließen
            ]
        }
    def get_description(self,row):
                return " ".join(str(row[col]) for col in ["PartDesc1", "PartDesc2", "PartDesc3", "PartDesc4"] if pd.notna(row[col]))

    def get_products(self):
        """
        Generates a dictionary of unique products from ``self.data``.

        This method extracts unique product entries from ``self.data`` and 
        formats them into a dictionary where each ``PartID`` serves as a key. 
        The corresponding value is a dictionary containing:
        
        - ``name``: The product ID (``PartID``).
        - ``desc``: A combined string of product descriptions (``PartDesc1-4``).

        Steps
        -----
        1. Filters unique ``PartID`` entries from ``self.data``.
        2. Constructs a dictionary containing product details.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary where each key is a ``PartID``, and the value is another 
            dictionary containing product name and description.

            Example output:
            
            .. code-block:: python
            
                {
                    "Item123": {
                        "name": "Item123",
                        "desc": "High-performance processor 8-core 3.5GHz"
                    },
                    "Item456": {
                        "name": "Item456",
                        "desc": "16GB DDR4 RAM high-speed memory"
                    }
                }

        Example
        -------
        >>> recommender = RecomBasket(data)
        >>> products = recommender.get_products()
        >>> print(products)
        """

        if self.data is None or self.data.empty:
            return {}

        # 🔹 1. EINZIGARTIGE ARTIKEL ERMITTELN
        unique_products = self.data.drop_duplicates(subset=["PartID"])

        # 🔹 2. PRODUKTLISTE ERSTELLEN
        products = {
            row["PartID"]: {
                "name": row["PartID"],
                "desc": " ".join(
                    str(row[col]) for col in ["PartDesc1", "PartDesc2", "PartDesc3", "PartDesc4"] if pd.notna(row[col])
                ).strip()
            }
            for _, row in unique_products.iterrows()
        }

        return products  