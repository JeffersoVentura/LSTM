from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model("modelo_lstm_stock.keras")  # Asegúrate que esté en la misma carpeta

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        ventana = np.array(data["ventana"]).reshape(1, 7, 5)
        stock_final_actual = ventana[0, -1, 3]
        umbral_stock = data.get("umbral_stock", 150)

        pred = model.predict(ventana)[0][0]

        if stock_final_actual > umbral_stock:
            es_sobrestock = 1
            prob_manual = 1.0
        else:
            es_sobrestock = int(pred > 0.5)
            prob_manual = pred

        consumo_recomendado = max(0, stock_final_actual - umbral_stock) if es_sobrestock else 0

        return jsonify({
            "probabilidad": float(round(prob_manual, 3)),
            "es_sobrestock": es_sobrestock,
            "consumo_recomendado": float(consumo_recomendado)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ⚠️ Aquí está el cambio importante
    app.run(host="0.0.0.0", port=port)
