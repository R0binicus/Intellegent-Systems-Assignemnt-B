# Intellegent-Systems-Assignemnt-B

## 1. Install Requirements
in cmd with the directory of inside the Intellegent-Systems-Assignemnt-B folder run:
pip install -r requirements.txt

this should install the requirements listed in the requirements.txt file


## 2. Prepare dataset
To download data, we provide 2 source, yahoo and tiingo (yahoo by default). We can read a list of stock market and run it. Example, we want to download and preprocess all stock market in tw50.csv with 20 period days and produce 50x50 image dimension.

```
$ python runallfromlist.py tw50.csv 20 50
```
Generate the final dataset. Example, we want to generate a final dataset from tw50 with 20 period days and 50 dimension.
```
$ python generatebigdata.py dataset 20_50 bigdata_20_50
```

## 3. Build the model
We can run build model with default parameter.
```
$ python myDeepCNN.py -i dataset/bigdata_20_50
```
