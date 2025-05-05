# mlflow_tracking/bentoml/service.py (Suggested modification)

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field
import pandas as pd
import logging
import traceback
import os # Import os to read environment variable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- (WineInput BaseModel remains the same) ---
class WineInput(BaseModel):
    data: list[list[float]] = Field(..., min_items=1)
    columns: list[str] = Field(
        default=[
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "Id"
        ],
        min_items=12,
        max_items=12
    )

    class Config:
        schema_extra = {
            "example": {
                "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]],
                "columns": [
                    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                    "pH", "sulphates", "alcohol", "Id"
                ]
            }
        }

svc = bentoml.Service("wine_quality_service")

# --- Load the model runner using the tag provided by the entrypoint ---
# This happens when the service definition is loaded/built by bentoml
BENTOML_MODEL_TAG = os.environ.get("BENTOML_SERVE_MODEL_TAG")

if not BENTOML_MODEL_TAG:
    # Fallback or raise error if the env var isn't set (should be set by entrypoint)
    logger.warning("BENTOML_SERVE_MODEL_TAG environment variable not set. Attempting to list models...")
    available_models = bentoml.models.list()
    if available_models:
        # Load the first available model as a fallback (less reliable)
        logger.warning(f"Using first available model as fallback: {available_models[0].tag}")
        model_to_serve = available_models[0]
    else:
        logger.error("No BentoML model tag provided via environment variable, and no models found in local store.")
        # Raise error during service loading if no model can be found
        raise ValueError("No BentoML model tag specified for serving.")
else:
    logger.info(f"Loading BentoML model with tag: {BENTOML_MODEL_TAG}")
    try:
        model_to_serve = bentoml.models.get(BENTOML_MODEL_TAG)
    except bentoml.exceptions.NotFound:
        logger.error(f"BentoML model with tag '{BENTOML_MODEL_TAG}' not found.")
        # Raise error during service loading if the specified model isn't found
        raise ValueError(f"BentoML model '{BENTOML_MODEL_TAG}' not found.")


# Create the runner from the loaded model
wine_runner = model_to_serve.to_runner()

# Add the runner to the BentoML service
svc.add_runner(wine_runner)


@svc.api(input=JSON(pydantic_model=WineInput), output=JSON())
async def predict(input_data: WineInput):
    try:
        input_df = pd.DataFrame(input_data.data, columns=input_data.columns)
        # Optional: Log input data - be cautious with sensitive data or high traffic
        # logger.info(f"Received input shape: {input_df.shape}")
        # logger.debug(f"Received input: {input_df.to_dict()}") # Use debug for potentially large inputs

        # Use the runner added to the service
        # The runner should be initialized by BentoML runtime
        predictions = await wine_runner.predict.async_run(input_df)

        logger.info(f"Generated {len(predictions)} predictions.")
        # logger.debug(f"Predictions: {predictions.tolist()}") # Use debug for potentially large outputs

        # Convert predicted labels (0, 1, 2) back to quality names ("low", "medium", "high")
        # This mapping needs to be accessible. You could store it as model metadata
        # or retrieve it from MLflow artifacts/params associated with the model.
        # For simplicity here, hardcode assuming the model always predicts 0, 1, or 2.
        # A more robust solution retrieves this mapping dynamically.
        quality_map = {0: "low", 1: "medium", 2: "high"}
        predicted_labels = [quality_map.get(int(p), "unknown") for p in predictions] # Handle potential non-integer predictions safely


        return {"predictions": predicted_labels} # Return labels as strings
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an error response with status code 500 (handled by BentoML by raising Exception)
        # Or return a JSON error payload and let BentoML handle status code
        # Returning a dict with 'error' key is a common pattern, but won't set HTTP status
        # To set status, raise web.HTTPException or similar depending on BentoML internal framework (Starlette)
        # For now, let BentoML raise the default 500 for uncaught exceptions
        raise # Re-raise the exception