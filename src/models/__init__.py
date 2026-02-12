from src.models.base import BaseModel
from src.models.logistic import LogisticModel
from src.models.xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "logistic": LogisticModel,
    "xgboost": XGBoostModel,
}
