# CYBER-CRIME-CLASSIFIER

## Repository Structure:
**BERT** - Content related to BERT language model 

**ROBERTA** - Content related to RoBERTa language model 

**HINGBERT** - Content related to HINGBERT language model 

**HINGROBERTA** - Content related to HINGRoBERTa language model

**INFERENCE.py** - An example script of inference the model

**PREPROCESSED.py** - Data preprocessign script




Download the augmented train and test dataset from this [link](https://drive.google.com/drive/folders/1CPupu7i7fgw_xC_a406qO_hxfpOUBt7j?usp=sharing) and place it in Folders, i.e., **BERT, ROBERTA, HINGBERT,** and **HINGROBERTA**.

Create a folder named **MODEL**

Download the model files from this [link](https://drive.google.com/drive/folders/1rlEs0p5KFJmMNWlQjMSk2oJ8OQrqkKa2?usp=sharing) and place the download files in **MODEL** folder.




tests/
├── __init__.py                    # Initializes the test package
├── test_main.py                   # Unit tests for main application logic
├── test_helper.py                 # Unit tests for helper functions
├── test_data_processing.py        # Unit tests for data processing functions
├── fixtures/                      # Reusable test data and setup configurations
│   ├── __init__.py
│   ├── test_data.json             # Sample test data
│   └── mock_config.yaml           # Mock configuration file
├── integration/                   # Integration tests for multiple components
│   ├── __init__.py
│   ├── test_api_integration.py    # Integration tests for API
│   └── test_module_integration.py # Integration tests for modules
├── performance/                   # Performance and load testing scripts
│   ├── __init__.py
│   └── test_performance.py        # Performance tests
└── test_utils.py                  # Utility functions for testing
