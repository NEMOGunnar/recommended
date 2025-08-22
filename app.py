
from flask import Flask, render_template, request, jsonify
from pattern.recom_basket import RecomBasket
from pattern.similar_parts import SimilarParts
import pandas as pd
from pathlib import Path

# Hole den Pfad des Hauptverzeichnisses (2 Ebenen hoch von main.py aus)
BASE_DIR = Path(__file__).resolve().parent

# Pfad zum "data"-Ordner
DATA_DIR = BASE_DIR / "data"

# Beispiel: Lade eine Datei aus dem data-Ordner
data_file = DATA_DIR / "sample_data.csv"

app = Flask(__name__)

filepath = DATA_DIR / "NemoDataExport.foc.gz"
print(filepath)
recom_basket = RecomBasket(filepath= filepath)
recom_basket.initialize()

sim_parts = SimilarParts(filepath= filepath)
sim_parts.init_similar_matrix()

# SPEICHERUNG DES WARENKORBS & BELEGE
shopping_cart = {}
order_history = []

@app.route("/")
def index():
    return render_template("index.html", products=recom_basket.get_products(), shopping_cart=shopping_cart)

@app.route("/update_cart", methods=["POST"])
def update_cart():
    """Aktualisiert den Warenkorb und gibt neue Empfehlungen zurück."""
    global shopping_cart

    data = request.get_json()
    shopping_cart = {k: int(v) for k, v in data.items() if int(v) > 0}
    print(shopping_cart)
    recommendations = recom_basket.recommend_articles(shopping_cart)

        # 🔹 ARTIKELBESCHREIBUNGEN ERMITTELN
    product_descriptions = {
        row["PartID"]: " ".join(
            str(row[col]) for col in ["PartDesc1", "PartDesc2", "PartDesc3", "PartDesc4"] if pd.notna(row[col])
        ).strip()
        for _, row in recom_basket.data.drop_duplicates(subset=["PartID"]).iterrows()
    }

    # 🔹 BESCHREIBUNGEN ZUM WARENKORB HINZUFÜGEN
    shopping_cart = {
        part_id: {"quantity": quantity, "description": product_descriptions.get(part_id, "Keine Beschreibung verfügbar")}
        for part_id, quantity in shopping_cart.items()
    }

    # 🔹 BESCHREIBUNGEN ZU EMPFEHLUNGEN HINZUFÜGEN
    recommendations = [
        {"part_id": rec[0], "score": rec[1], "description": product_descriptions.get(rec[0], "Keine Beschreibung verfügbar")}
        for rec in recommendations
    ]

    return jsonify({"cart": shopping_cart, "recommendations": recommendations, "order_history": order_history})

@app.route("/update_orders", methods=["POST"])
def update_orders():
    """Aktualisiert die Liste der Bestellungen basierend auf dem Warenkorb."""
    global shopping_cart
    data = request.get_json()

    # 🔹 1. Falls keine Daten kommen, gib leere Antwort zurück
    if not data:
        return jsonify({"orders": {}})

    # 🔹 2. Konvertiere die Mengen in Integer und entferne ungültige Werte
    shopping_cart = {k: int(v) for k, v in data.items() if str(v).isdigit() and int(v) > 0}

    print("📌 Eingehender Warenkorb:", shopping_cart)  # DEBUG

    # 🔹 3. Bestellungen abrufen
    orders = recom_basket.get_order_history_from_cart(shopping_cart)

    print("📌 Generierte Bestellungen:", orders)  # DEBUG

    # 🔹 4. JSON-Antwort zurückgeben
    return jsonify({"orders": orders})

@app.route("/settings")
def settings():
    return render_template("settings.html")



@app.route("/sim_parts")
def home():
    """Hauptseite mit HTML-Template für Part Comparison"""
    products = sim_parts.get_products()
    print(products)
    return render_template("comp.html", products=products)

@app.route("/get_similar_parts", methods=["POST"])
def get_similar_parts():
    selected_parts = request.json  # Liste der gewählten PartIDs
    result = sim_parts.get_similar_parts(selected_parts)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)