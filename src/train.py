import sys
from src.model import Model

if __name__=="__main__":
    if len(sys.argv) !=2 :
        print("Please provide required parameters")

    data_path = sys.argv[1]
    tmodel = Model()
    tmodel.load_data(data_path)
    tmodel.character_model()