from src.models.base import BaseModel
from src.models.logistic import LogisticModel
from src.models.xgboost_model import XGBoostModel
from src.models.stacking import StackingModel

MODEL_REGISTRY = {
    "logistic": LogisticModel,
    "xgboost": XGBoostModel,
    "stacking": StackingModel,
}
