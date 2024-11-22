# CYBER-CRIME-CLASSIFIER


## Description of Folders and Files
- **`Language Model Implmentation`**: Contains the implementation of all Language Models
  - **`BERT`** - Content related to BERT language model
  - **`ROBERTA`** - Content related to RoBERTa language model
  - **`HINGBERT`** - Content related to HINGBERT language model
  - **`HINGROBERTA`** - Content related to HINGRoBERTa language model
- **`ML Implmentation`** - Contains the implementation of all Machine Learning classifiers
  - **`ADABOOST.py`** - AdaBoost classifier implementation
  - **`KNeighbors.py`** - K-nearest neighbors classifier implementation
  - **`RANDOMFOREST.py`** - Random Forest classifier implementation
  - **`XGBOOST.py`** - XGBoost classifier implementation
- **`AUGMENTATION.py`** - Data augmentation script
- **`INFERENCE.py`** - An example script of inference the model
- **`PREPROCESSED.py`** - Data preprocessing script


## Run Steps
- Download the augmented train and test dataset from this [link](https://drive.google.com/drive/folders/1CPupu7i7fgw_xC_a406qO_hxfpOUBt7j?usp=sharing) and place it in Folders, i.e., **ML Implmentation**, **BERT, ROBERTA, HINGBERT,** and **HINGROBERTA**
- Run `{model_name}.py` present in each folders , i.e., **ML Implmentation**, **BERT, ROBERTA, HINGBERT,** and **HINGROBERTA**.

``` choices for {model_name}: 'BERT', 'ROBERTA', 'HINGBERT', 'HINGROBERTA', 'ADABOOST', 'KNeighbors', 'RANDOMFOREST', 'XGBOOST' ```

## Inference
- Create a folder named **MODEL**
- Download the model files from this [link](https://drive.google.com/drive/folders/1rlEs0p5KFJmMNWlQjMSk2oJ8OQrqkKa2?usp=sharing) and place the download files in **MODEL** folder
- Run `INFERENCE.py` (Keep in mind that to replace `example_text` value in the script with respective crime info )


