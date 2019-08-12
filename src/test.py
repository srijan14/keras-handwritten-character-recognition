import sys
from src.model import Model

if __name__=="__main__":
    if len(sys.argv) !=2 :
        print("Please provide required parameters. Example run\npython src/test.py  models/model.01-1.1955.hdf5 data/test/s.jpg")

    if len(sys.argv) == 3:
        model_path = sys.argv[1]
        img_path = sys.argv[2]
    elif len(sys.argv) == 2:
        img_path = None
        model_path = sys.argv[1]
    else:
        img_path = None
        model_path = None


    test_model = Model()
    test_model.loadmodel(path=model_path)
    test_model.test(img_path=img_path)
