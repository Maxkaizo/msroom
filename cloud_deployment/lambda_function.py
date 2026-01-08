
import pickle
import json
import os
import numpy as np
import pandas as pd

# Global variable to cache the model (warm starts)
model_dict = None

def load_model():
    """Load trained model and encoders"""
    global model_dict
    
    # In Lambda container, we'll put the model in the standard location
    # The Dockerfile will copy models/ to ${LAMBDA_TASK_ROOT}/models/
    model_path = "models/model.pkl"

    if model_dict is not None:
        return True

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False

    try:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def lambda_handler(event, context):
    """
    AWS Lambda Handler
    """
    global model_dict
    
    # Ensure model is loaded
    if model_dict is None:
        success = load_model()
        if not success:
             return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to load model'})
            }

    try:
        # Parse logic: Handle both direct invocation and API Gateway proxy integration
        body = event
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        
        # Extract model components
        model = model_dict["model"]
        ohe = model_dict["ohe"]
        le_target = model_dict["le_target"]
        
        # Feature processing logic (copied from predict.py)
        # We need to ensure the input body has the expected keys
        # We'll use the same list of columns as predict.py
        
        categorical_cols = [
            "cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed",
            "gill-attachment", "gill-spacing", "gill-color", "stem-surface",
            "stem-color", "has-ring", "ring-type", "habitat", "season"
        ]

        numerical_cols = [
            "cap-diameter", "stem-height", "stem-width", "spore_print_color_present"
        ]
        
        # Prepare input data
        input_data = {}
        for col in categorical_cols:
            input_data[col] = [body.get(col)]

        for col in numerical_cols:
            if col == "spore_print_color_present":
                input_data[col] = [body.get(col, 0)] # Default to 0
            else:
                input_data[col] = [body.get(col)]
                
        df_input = pd.DataFrame(input_data)
        
        # Apply OneHotEncoding
        X_cat_encoded = ohe.transform(df_input[categorical_cols])
        
        # Combine with numerical features
        X_encoded = np.hstack([X_cat_encoded, df_input[numerical_cols].values])
        
        # Make prediction
        prediction_proba = model.predict_proba(X_encoded)[0]
        prediction_class = model.predict(X_encoded)[0]
        
        # Map prediction to class name
        predicted_class_name = le_target.inverse_transform([prediction_class])[0]
        confidence = float(prediction_proba[prediction_class])
        
        prediction_label = "edible" if predicted_class_name == "e" else "poisonous"
        
        response = {
            "prediction": prediction_label,
            "probability": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.2f}%"
        }
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(response)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
