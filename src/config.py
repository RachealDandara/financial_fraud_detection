"""Central place for paths and project-wide settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

TARGET = "Class"

# Business assumption: cost of a false alarm (blocked/queried legitimate
# transaction: support call, customer friction). Change it and re-run to see
# how the optimal decision threshold moves.
FALSE_ALARM_COST = 10.0

RANDOM_STATE = 42
