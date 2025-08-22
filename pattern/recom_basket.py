import numpy as np
import pandas as pd
import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity

class RecomBasket:
    
    def __init__(self, load_last_data = False, filepath=""):
        if load_last_data == False:
            self.read_gzip(filepath)
            self.min_threshold = 0.1

    def normalize_to_identity(self,A):
        """
        Normalizes a square matrix A such that A @ A.T = I (orthogonal matrix).

        This method applies Singular Value Decomposition (SVD) to transform 
        the given matrix into an orthogonal matrix, ensuring that its 
        transpose times itself results in an identity matrix.

        Steps:
        -------
        1. Performs Singular Value Decomposition (SVD) on matrix A.
        2. Computes an orthogonal matrix by multiplying U and V^T.

        Parameters:
        -----------
        A : np.ndarray
            A square matrix (NxN) to be normalized.

        Returns:
        --------
        np.ndarray
            The normalized orthogonal matrix where A @ A.T ≈ I.

        Example:
        --------
        A = np.array([[3, 1], [2, 4]])
        recommender = RecomBasket()
        A_norm = recommender.normalize_to_identity(A)
        print(A_norm)

        # Checking if the result is approximately an identity matrix:
        print(np.round(A_norm @ A_norm.T, 5))
        """
        U, _, Vt = np.linalg.svd(A)  # SVD-Zerlegung
        A_norm = U @ Vt  # Erzeugt eine orthogonale Matrix

        return A_norm
    
    def read_gzip(self, file_path):
        """
        Reads a compressed gzip CSV file and loads the relevant columns into a pandas DataFrame.

        This method reads a `.gzip`-compressed CSV file, extracts specified columns, 
        and filters out rows where `PartID` is missing. The first 100 rows are then saved 
        to a CSV file with a timestamped filename.

        Steps:
        -------
        1. Reads the gzip-compressed CSV file.
        2. Extracts relevant columns from the dataset.
        3. Removes rows where `PartID` is missing.
        4. Saves the first 100 rows to a timestamped CSV file.
        5. Stores the processed DataFrame in `self.data`.

        Parameters:
        -----------
        file_path : str
            The path to the gzip-compressed CSV file.

        Returns:
        --------
        None (updates internal attribute `self.data`)

        Attributes Updated:
        -------------------
        data : pd.DataFrame
            A pandas DataFrame containing filtered and processed order data.

            Example DataFrame:
            ```
            +-----------+-----------+-----------+-----------+--------+-----------------+------------+---------------+
            | PartDesc1 | PartDesc2 | PartDesc3 | PartDesc4 | PartID | OrderDocLineQty | OrderDocID | OrderDocLineID |
            +-----------+-----------+-----------+-----------+--------+-----------------+------------+---------------+
            | Item A    | Type X    | Size M    | Color Red | 12345  | 10              | 56789      | 987654321      |
            | Item B    | Type Y    | Size L    | Color Blue| 67890  | 5               | 98765      | 123456789      |
            +-----------+-----------+-----------+-----------+--------+-----------------+------------+---------------+
            ```

        Example:
        --------
        file_path = "data/orders.csv.gz"
        self.read_gzip(file_path)
        print(self.data.head())
        """
        df = pd.read_csv(file_path, compression="gzip",sep= ";",encoding="utf-8",nrows=5000 ,
                 usecols=[
                        "PartDesc1",
                        "PartDesc2",
                        "PartDesc3",
                        "PartDesc4", 
                        "PartID",
                        "OrderDocLineQty",
                        "OrderDocID",
                        "OrderDocLineID"
                        ]).dropna(subset=[
                        "PartID" 
                                          ]) #.drop_duplicates(subset=["PartID"])

    
        # Speichern der ersten 100 Zeilen in eine CSV-Datei
        output_file = "data/Belege.csv"
        time_stamp_file = self.add_timestamp_to_filename(output_file)
        #df.to_csv(time_stamp_file, index=False, sep=";")
        self.data = df
        print(f"gespeichert in: {time_stamp_file}")  

    def add_timestamp_to_filename(self,file_path):
        """
        Appends a timestamp to the filename before the file extension.

        This method takes a file path, extracts the file name and extension, 
        and inserts a timestamp in the format `YYYYMMDD_HHMMSS` between them. 
        If the file is located in a directory, the directory path is preserved.

        Steps:
        -------
        1. Retrieves the current date and time.
        2. Extracts the file name and extension.
        3. Inserts the timestamp before the file extension.
        4. Returns the new file path.

        Parameters:
        -----------
        file_path : str
            The original file path, including the file name and extension.

        Returns:
        --------
        str
            A new file path with a timestamp added.

            Example output:
            - Input: `"data.csv"`
            - Output: `"data_20240204_153045.csv"`

        Example:
        --------
        file_path = "report.xlsx"
        new_file_path = self.add_timestamp_to_filename(file_path)
        print(new_file_path)  
        # Output: "report_20240204_153045.xlsx"
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

    def create_order_matrix(self):
        """
        Creates an order matrix where each row represents an order ("OrderDocID") 
        and each column represents a product ("PartID"). The values indicate how 
        many units of a product are present in an order.

        This method constructs a structured matrix representation of historical 
        order data, which serves as the foundation for recommendation calculations.

        Steps:
        -------
        1. Identifies unique "PartID"s (columns) and "OrderDocID"s (rows).
        2. Maps "PartID"s to column indices and "OrderDocID"s to row indices.
        3. Initializes an empty matrix ("MxN") with zeros.
        4. Fills the matrix with order quantities from the dataset.
        5. Stores the "PartID" and "OrderDocID" mappings for future use.

        Parameters:
        -----------
        None (uses "self.data" internally, which must contain the following columns: 
        ["OrderDocID", "PartID", "OrderDocLineQty"])

        Returns:
        --------
        None (updates internal attributes)

        Attributes Updated:
        -------------------
        order_matrix : np.ndarray
            An "MxN" matrix where "M" is the number of orders and "N" is the number 
            of unique products. The values represent the quantity of each product 
            in a given order.

        part_id_mapping : list
            A list of unique "PartID"s, representing the column order in the matrix.

        order_id_mapping : list
            A list of unique "OrderDocID"s, representing the row order in the matrix.

        Example:
        --------
        recommender = RecomBasket(data)
        recommender.create_order_matrix()
        print(recommender.order_matrix)
        print(recommender.part_id_mapping)
        print(recommender.order_id_mapping)
        """
        df = self.data
        # 🔹 1. EINZIGARTIGE PARTID & ORDERDOCID ERMITTELN
        unique_part_ids = df["PartID"].unique()
        unique_order_ids = df["OrderDocID"].unique()

        # Anzahl der Spalten (N) und Reihen (M)
        N = len(unique_part_ids)
        M = len(unique_order_ids)

        # 🔹 2. MAPPING: PartID → Spaltenindex
        part_id_to_index = {part_id: idx for idx, part_id in enumerate(unique_part_ids)}

        # 🔹 3. MAPPING: OrderDocID → Zeilenindex
        order_id_to_index = {order_id: idx for idx, order_id in enumerate(unique_order_ids)}

        # 🔹 4. INITIALISIERE DIE MATRIX MIT NULLEN (MxN)
        order_matrix = np.zeros((M, N))

        # 🔹 5. DATEN IN DIE MATRIX EINTRAGEN
        for _, row in df.iterrows():
            order_idx = order_id_to_index[row["OrderDocID"]]
            part_idx = part_id_to_index[row["PartID"]]
            order_matrix[order_idx, part_idx] += pd.to_numeric(row["OrderDocLineQty"], errors="coerce") if pd.notna(row["OrderDocLineQty"]) else 0 #order_matrix[order_idx, part_idx] += row["OrderDocLineQty"]  # Addiert Mengen pro Order

        # 🔹 6. SPEICHERE DAS PARTID-MAPPING FÜR SPALTENZUGEHÖRIGKEIT
        self.part_id_mapping = list(unique_part_ids)
        self.order_id_mapping = list(unique_order_ids)
        self.order_matrix = order_matrix

    
    def compute_correlation_matrix(self):
        """
        Computes the correlation matrix by multiplying the given order-part matrix 
        with its transposed version.

        Formula:
        --------
        C = Aᵀ * A

        This method calculates an item-item correlation matrix based on order 
        co-occurrence, then normalizes it to maintain identity-like scaling.

        Steps:
        -------
        1. Computes the product of the transposed order matrix and the order matrix itself.
        2. Normalizes the resulting correlation matrix.
        3. Rounds the final matrix to 4 decimal places.

        Parameters:
        -----------
        None (uses `self.order_matrix` internally)

        Returns:
        --------
        None (updates `self.correlation_matrix_norm`)

        Attributes Updated:
        -------------------
        correlation_matrix_norm : np.ndarray
            An N x N correlation matrix representing the relationship between items.

            Example:
            ```
            [[1.00, 0.78, 0.34],
            [0.78, 1.00, 0.56],
            [0.34, 0.56, 1.00]]
            ```

        Example:
        --------
        recommender = RecomBasket(order_matrix)
        recommender.compute_correlation_matrix()
        print(recommender.correlation_matrix_norm)
        """
        
        # 🔹 1. TRANSFORMATION: A^T * A and normalize
        self.correlation_matrix_norm = np.round(self.normalize_to_identity(np.dot(self.order_matrix.T, self.order_matrix)),4)

    def compute_similarity_matrix(self):
        """
        Computes a Cosine Similarity Matrix (values between 0 and 1) instead of AᵀA 
        to avoid negative values.

        This method calculates the cosine similarity between products based on 
        their co-occurrence in past orders. It ensures that negative values 
        are set to 0, keeping similarity scores between 0 and 1.

        Steps:
        -------
        1. Computes cosine similarity for the order-part matrix (column-wise).
        2. Ensures that no negative values exist by setting them to 0.
        3. Rounds the similarity values to 3 decimal places.

        Parameters:
        -----------
        None (uses `self.order_matrix` internally)

        Returns:
        --------
        None (updates `self.similarity_matrix`)

        Attributes Updated:
        -------------------
        similarity_matrix : np.ndarray
            An N x N matrix containing similarity scores between items (range: 0 to 1).
            
            Example:
            ```
            [[1.00, 0.85, 0.32],
            [0.85, 1.00, 0.45],
            [0.32, 0.45, 1.00]]
            ```

        Example:
        --------
        recommender = RecomBasket(order_matrix)
        recommender.compute_similarity_matrix()
        print(recommender.similarity_matrix)
        """

        # Berechnung der Cosine Similarity
        similarity_matrix = cosine_similarity(self.order_matrix.T)  # Spaltenweise Ähnlichkeit

        # Negative Werte auf 0 setzen
        similarity_matrix = np.maximum(similarity_matrix, 0)

        self.similarity_matrix = np.round(similarity_matrix,3)

    def recommend_articles(self,shopping_list):
        """
        Computes product recommendations based on a shopping list and a Cosine Similarity Matrix.

        This method takes a shopping list, converts it into a vector aligned with ``part_id_mapping``,
        multiplies it with a precomputed similarity matrix, and returns the top 10 recommended products.

        Steps
        -----
        1. Creates a shopping list vector aligned with ``part_id_mapping``.
        2. Multiplies this vector with the normalized similarity matrix.
        3. Rounds the result to 1 decimal place.
        4. Filters items below a predefined threshold (``min_threshold``).
        5. Removes items already present in the shopping list.
        6. Sorts recommendations by quantity (highest first).
        7. Returns the top 10 recommended products.

        Parameters
        ----------
        shopping_list : dict
            A dictionary containing ``{"PartID": quantity}``.

            Example:

            .. code-block:: python

                {
                    "Item123": 2,
                    "Item456": 1
                }

        Returns
        -------
        list
            A list of the top 10 recommended items, each represented as ``(PartID, quantity)``.

            Example output:

            .. code-block:: python

                [
                    ("Item789", 5.3),
                    ("Item321", 4.7),
                    ("Item654", 3.9)
                ]

        Example
        -------
        >>> shopping_list = {"Item123": 2, "Item456": 1}
        >>> recommendations = self.recommend_articles(shopping_list)
        >>> print(recommendations)
        """

        # 🔹 1. INITIALISIERE DEN VEKTOR MIT NULLEN
        shopping_vector = np.zeros(len(self.part_id_mapping))

        # 🔹 2. SETZE DIE MENGEN IN DEN VEKTOR EIN
        for part_id, quantity in shopping_list.items():
            if part_id in self.part_id_mapping:
                idx = self.part_id_mapping.index(part_id)
                shopping_vector[idx] = quantity  # Setze die Menge an die richtige Stelle

        # 🔹 3. MULTIPLIZIERE DEN VEKTOR MIT DER COSINE-SIMILARITY-MATRIX
        result_vector = np.dot(self.similarity_matrix, shopping_vector)

        # 🔹 4. RUNDE DIE ERGEBNISSE AUF 1 NACHKOMMASTELLE
        result_vector = np.round(result_vector, 1)

        # 🔹 5. ERSTELLE DIE LISTE DER ARTIKEL MIT DER ERGEBNISMENGE
        recommendations = [(self.part_id_mapping[i], result_vector[i]) for i in range(len(result_vector))]

        ## 🔹 6. FILTERE ARTIKEL, DIE UNTER DER SCHWELLE LIEGEN
        recommendations = [item for item in recommendations if item[1] >= self.min_threshold]

        # 🔹 7. ENTFERNE ARTIKEL, DIE BEREITS IM WARENKORB SIND
        recommendations = [item for item in recommendations if item[0] not in shopping_list]

        # 🔹 8. SORTIERE NACH MENGE (HÖCHSTE ZUERST)
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

        # 🔹 9. GIB DIE TOP 10 ARTIKEL ZURÜCK
        print('call get recommendations')
        return recommendations[:10]  

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
    
    def initialize(self):
        """
        Initializes the recommendation system by generating the order matrix 
        and computing the similarity matrix.

        This method performs two key steps:
        1. Calls `create_order_matrix()` to build a structured dataset of orders.
        2. Calls `compute_similarity_matrix()` to calculate item-based similarities 
        for recommendation purposes.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
            This method does not return any value but initializes internal data structures.

        Example:
        --------
        recommender = RecomBasket(data)
        recommender.initialize()
        """
        self.create_order_matrix()
        self.compute_similarity_matrix()

    def get_order_history_from_cart(self, shopping_cart):
        """
        Generates a list of orders based on items present in the shopping cart.

        This method filters `self.data` to retain only rows with items in ``shopping_cart``, 
        retrieves relevant order IDs (``OrderDocID``), includes all associated items from those orders, 
        and formats the data into a structured dictionary.

        Steps
        -----
        1. Filters ``self.data`` to retain only rows with items from ``shopping_cart``.
        2. Retrieves all associated order IDs (``OrderDocID``) for the filtered items.
        3. Includes all items from these ``OrderDocID`` orders.
        4. Combines item descriptions (``PartDesc1-4``) into a single field.
        5. Converts ``OrderDocLineQty`` to `int`, replacing `NaN` with `0`.
        6. Groups results by ``OrderDocID``.

        Parameters
        ----------
        shopping_cart : dict
            Dictionary containing item IDs as keys and quantities as values.

            Example format:
            
            .. code-block:: python
            
                {
                    "Item123": 2,
                    "Item456": 1
                }

        Returns
        -------
        dict
            A dictionary where keys are ``OrderDocID`` and values are lists of order details.

            Example output format:
            
            .. code-block:: python

                {
                    "OrderDocID1": [
                        {"PartID": "Item123", "OrderDocLineQty": 2, "Description": "Item A Description"},
                        {"PartID": "Item789", "OrderDocLineQty": 5, "Description": "Item B Description"}
                    ],
                    "OrderDocID2": [
                        {"PartID": "Item456", "OrderDocLineQty": 1, "Description": "Item C Description"}
                    ]
                }

        Example
        -------
        >>> shopping_cart = {"Item123": 2, "Item456": 1}
        >>> order_history = self.get_order_history_from_cart(shopping_cart)
        >>> print(order_history)
        """

        if self.data is None or self.data.empty:
            return {}

        # 🔹 1. ARTIKEL FILTERN, DIE IM WARENKORB SIND
        filtered_orders = self.data[self.data["PartID"].isin(shopping_cart.keys())]

        # 🔹 2. ALLE EINZIGARTIGEN OrderDocIDs FINDEN
        relevant_order_ids = filtered_orders["OrderDocID"].unique()

        # 🔹 3. HOLEN ALLER ZUGEHÖRIGEN BESTELLUNGEN (ALLE POSITIONEN)
        orders_data = self.data[self.data["OrderDocID"].isin(relevant_order_ids)].copy()

        # 🔹 4. BESCHREIBUNG ZUSAMMENSETZEN (PartDesc1-4)
        orders_data["Description"] = orders_data[["PartDesc1", "PartDesc2", "PartDesc3", "PartDesc4"]].astype(str).agg(' '.join, axis=1).str.strip()

        # 🔹 5. NaN-WERTE IN `OrderDocLineQty` ERSETZEN UND IN `int` UMWANDELN
        orders_data["OrderDocLineQty"] = pd.to_numeric(orders_data["OrderDocLineQty"], errors="coerce").fillna(0).astype(int)

        # 🔹 6. RELEVANTE FELDER BEHALTEN
        orders_data = orders_data[["OrderDocID", "PartID", "OrderDocLineQty", "Description"]]

        # 🔹 7. BESTELLUNGEN NACH OrderDocID GRUPPIEREN
        grouped_orders = orders_data.groupby("OrderDocID").apply(lambda x: x.to_dict(orient="records")).to_dict()
        print('call get grouped orders')
        return grouped_orders       