# Handwritten-Character-Recognition
This repository performs character recognition on  [EMINST](https://www.nist.gov/node/1298471/emnist-dataset) dataset.

## Dependencies
1. Python2,Python3

## Installing Requirements
    pip install -r requirements.txt

## Pretrained Models

Pretrained Model trained on eminst by class dataset is already present inside the *models* folder

## Train

   Put the .mat file downloaded from the EMINST page inside data folder.
   
   Training Parameters can be changed inside the **src/constants.py**
   
   Also the model architecture can be changes from inside **src/define_mode.py**
   
    python src/train.py --data ./data/emnist-byclass.mat
    
    
## Prediction

    python src/test.py --data ./data/test/s.jpg --model ./models/model.hdf5'