from pathlib import Path

# CONSTANTS
# constants.py lives in src; its parent folder is the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_FOLDER = BASE_DIR / "datasets"
OUTPUTS_FOLDER = BASE_DIR / "outputs"

N_EXPECTED_BATCHES = 30
DOWNTIME_D = 1.0  # Downtime between batches [d]

# Store column names once to reduce spelling mistakes.
BATCH_ID = "Batch ID"
TIME_H = "Time [h]"
TIME_D = "Time [d]"

GLUCOSE = "Glucose [mM]"
GLUTAMINE = "Glutamine [mM]"
VCD = "VCD [cells/mL]"
TCD = "TCD [cells/mL]"
LACTATE = "Lactate [mM]"
AMMONIA = "Ammonia [mM]"
MAB = "mAb [mg/L]"

INITIAL_GLUCOSE = "Initial Glucose [mM]"
INITIAL_GLUTAMINE = "Initial Glutamine [mM]"
INITIAL_VCD = "Initial VCD [cells/mL]"

COST = "Cost [$/L]"
VALUE = "Value [$/L]"
PROFIT = "Profit [$/L]"
SPACE_TIME_PROFIT = "Space-time profit [$/(L*d)]"
SPACE_TIME_YIELD = "Space-time yield [mg/(L*d)]"

PROCESS_COLUMNS = [
    GLUCOSE, GLUTAMINE, VCD, TCD, LACTATE, AMMONIA, MAB
]
# MODEL VALIDATION
RANDOM_STATE = 0
TEST_FRACTION = 0.20
N_CV_SPLITS = 4