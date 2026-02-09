# Training Hyperparameters
INPUT_SIZE = 224
NUM_CLASSES = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
NUM_EPOCHS = 500
MONITOR = 'Val/ACC'
MONITOR_MODE = 'max'
EARLYSTOP_PATIENCE = 50

# Model Hyperparameters
PROJECT_NAME = "MultiModeModel_Class{}_v3".format(NUM_CLASSES)
MODEL_NAME = "TX"
MODE = MODEL_NAME
CHECKPOINT_DIR = "checkpoints_Class{}_v3".format(NUM_CLASSES)
CHECKPOINT_NAME = MODEL_NAME+'_best'
SINGLE_DECODER = True
NEED_CL = True
if NUM_CLASSES == 2:
    ALPHA = [0.5 ,0.5]
elif NUM_CLASSES == 3:
    ALPHA = [0.12, 0.18, 0.7]
GAMMA = 2
DELTA = 0.7

# Dataset Hyperparameters
DATA_DIR = "Dataset_Class"+str(NUM_CLASSES)
NUM_WORKERS = 4

# Compute Hyperparameters
ACCELERATOR = "gpu"
DEVICES = [7]
PRECISION = 32
