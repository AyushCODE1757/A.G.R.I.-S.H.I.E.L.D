# Quick test in a temporary script or terminal:
import sys
from app.core.logger import logging
from app.core.exceptions import CustomException

try:
    logging.info("A.G.R.I.-S.H.I.E.L.D. infrastructure test initialized.")
    x = 1 / 0
except Exception as e:
    logging.error("Test exception triggered.")
    raise CustomException(e, sys)