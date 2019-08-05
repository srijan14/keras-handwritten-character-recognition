import sys
from src.model import Model

if __name__=="__main__":
    if len(sys.argv) !=2 :
        print("Please provide required parameters")

    if len(sys.argv) == 2:
        model_path = sys.argv[1]
    else:
        model_path = None

    test_model = Model()
    test_model.loadmodel()
    test_model.test()