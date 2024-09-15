# app/models.py

from pydantic import BaseModel
from typing import List

class TrainRequest(BaseModel):
    n_clusters: int = 3

class TrainResponse(BaseModel):
    message: str
    avg_centers: List[List[float]]

class GlobalModelResponse(BaseModel):
    avg_centers: List[List[float]]

class ErrorResponse(BaseModel):
    detail: str
