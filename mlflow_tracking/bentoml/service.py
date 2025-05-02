import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field
import pandas as pd
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@svc.api(input=JSON(pydantic_model=WineInput), output=JSON())
async def predict(input_data: WineInput):
    try:
        input_df = pd.DataFrame(input_data.data, columns=input_data.columns)
        logger.info(f"Received input: {input_df.to_dict()}")

        # Dynamically select a runner (first available model)
        available_models = bentoml.models.list()
        if not available_models:
            raise ValueError("No BentoML models available")
        model = available_models[0]
        runner = model.to_runner()
        await runner.init_local()

        predictions = await runner.predict.async_run(input_df)
        logger.info(f"Predictions: {predictions.tolist()}")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}