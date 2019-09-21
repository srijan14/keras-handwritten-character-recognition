import os


CHECKPOINT_PATH = "./models/model.{epoch:02d}-{val_loss:.4f}.hdf5"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
LOG_FILE = "./logs/training.log"

#Training Data Path
DATA_PATH = "./data/emnist-byclass.mat"

#Default Prediction File Path
TEST_PATH = "./data/test/test.png"

#Stop the training process if the validation accuracy don't improve for 10 continuoes epochs.
EARLY_STOP_PATIENCE = 10

#Batch size
BATCH_SIZE = 128

#Number of epochs
EPOCH = 1

# Change below parameter to true, to resume training
LOAD_MODEL = False
LOAD_MODEL_NAME = "./models/model.hdf5"

#Total Number of prediction classed
NUM_CLASSES = 62

