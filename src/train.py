import sys
from src.model import Model

if __name__=="__main__":
    if len(sys.argv) !=2 :
        print("Please provide required parameters. Training data path not provided. Example Run \n python src/train.py data/emnist-byclass.mat")
        exit()
    data_path = sys.argv[1]
    train_model = Model()
    train_model.load_data(data_path)
    train_model.character_model()
    train_model.train()
