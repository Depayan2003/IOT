import pandas as pd
from app.model_loader import model, label_encoder


def predict_quality(data):
    input_df = pd.DataFrame([{
        "fixed acidity": data.fixed_acidity,
        "volatile acidity": data.volatile_acidity,
        "citric acid": data.citric_acid,
        "residual sugar": data.residual_sugar,
        "chlorides": data.chlorides,
        "free sulfur dioxide": data.free_sulfur_dioxide,
        "total sulfur dioxide": data.total_sulfur_dioxide,
        "density": data.density,
        "pH": data.ph,
        "sulphates": data.sulphates,
        "alcohol": data.alcohol
    }])

    pred_encoded = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(input_df)[0]
    confidence = {
        label_encoder.classes_[i]: float(round(probs[i], 4))
        for i in range(len(probs))
    }

    return {
        "predicted_quality": pred_label,
        "confidence": confidence
    }