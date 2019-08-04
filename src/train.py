import sys
from src.model import Model

if __name__=="__main__":
    if len(sys.argv) !=2 :
        print("Please provide required parameters")

    data_path = sys.argv[1]
    train_model = Model()
    train_model.load_data(data_path)
    train_model.character_model()
    train_model.train()