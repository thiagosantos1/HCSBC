# HCSBC: Hierarchical Classification System for Breast Cancer Specimen Report

<!-- TOC --> 

- [Instruction Navigation](#information-extraction-from-breast-cancer-pathology-reports)
    - [Publication](#publication)
    - [Web System Application](#web-system-application)
    - [Model and Project Design](#model-and-project-design)
    - [Installation](#installation)
    - [Download Models](#download-models)
    - [Demo App](#demo-app)
    - [Extract Using Terminal](#extract-using-terminal)
    - [Annotation Tool](#annotation-tool)
    - [Contributors](#contributors)
    - [References](#references)

<!-- /TOC -->


## Publication

This is the code repository for the paper: In Progress
	
## Web System Application 

We developed a web system application for users to test our proposed pipilne for predicting histopathology reports. Users can interact with the platform in 2 ways: 1) Input an excel/csv spreadsheet with a column with the biopsy diagnosis (Part A,B or C). 2) Input a single biopsy diagnosis. An example of our Web System is illustraded bellow:

<table border=1>
<tr align='center' > 
<td><img src="https://github.com/thiagosantos1/HCSBC/blob/main/app/imgs/app_1.png" width="500"                  title="HCS App"></td>         
<td><img src="https://github.com/thiagosantos1/HCSBC/blob/main/app/imgs/app_2.png" width="500" title="HCS App"></td>
</tr>
</table>

## Model and Project Design
<table border=1>
<tr align='center' > 
<td><img src="https://github.com/thiagosantos1/HCSBC/blob/main/app/imgs/pipeline.png" width="500"                  title="HCS App"></td>         
<td><img src="https://github.com/thiagosantos1/HCSBC/blob/main/app/imgs/hybrid_system.png" width="400" title="HCS App"></td>
</tr>
</table>




## Installation

We recommend using a virtual environment

If you do not already have `conda` installed, you can install Miniconda from [this link](https://docs.conda.io/en/latest/miniconda.html#linux-installers) (~450Mb). Then, check that conda is up to date:

```bash
conda update -n base -c defaults conda
```

And create a conda environment from the yml file:

```bash
conda env create -f environment.yml
```

If not already activated, activate the new conda environment using:

```bash
conda activate pathology
```


### Mac Users - Install libomp 11.0

```bash
wget https://raw.githubusercontent.com/chenrui333/homebrew-core/0094d1513ce9e2e85e07443b8b5930ad298aad91/Formula/libomp.rb
```

```bash
brew unlink libomp
```

```bash
brew install --build-from-source ./libomp.rb
```

#### Check that version 11.1 is installed 

```bash
brew list --version libomp
```

### Download Models

Script Example to Download Models
```bash
python3 app/src/download_models.py --all_labels "single_tfidf" --higher_order "PathologyEmoryPubMedBERT"
```


There are several models available for download

|Higher Order Option|All Labels Options           |
|------------|-----------------------------|
|PathologyEmoryPubMedBERT   | single_tfidf               |
|PathologyEmoryBERT  | branch_tfidf|
|ClinicalBERT        |                          |
|BlueBERT  |                   |
|BioBERT  |                   |
|BERT  |                   |


## Demo app

A minimal demo app is provided for you to play with the classification model!

<table border=1>
<tr align='center' > 
<td><img src="https://github.com/thiagosantos1/HCSBC/blob/main/app/imgs/app_1.png" width="500"                  title="HCS App"></td>         
</table>

You can easily run your app in your default browser by running:

```shell
python3.8 -m streamlit.cli run app/src/app.py
```
OR
```shell
streamlit run app/src/app.py
```

## Extract Using Terminal

You can also use our api to run using terminal.

The program takes an excell/csv sheet and extract the higher order and cancer characteristics from pathology reports 

- Input Options:
    - path_to_file - Path to an excel/csv with pathology diagnosis: String (Required).
    - column_name - Which column has the pathology diagnosis: String (Required).
    - higher_model - Which version of higher order model to use: String (Required).
    - all_label_model - Which version of all labels model to use: String (Required).
    - save_predictions - Path to save output: String (Optional).
    - output_model_data - Option to output model data to csv True/False (Optional).
    - save_input - Option to output the input fields True/False (Optional).
    - save_json - Path to save json analyis: String (Optional).

Example of Runing:
```shell
python3 app/src/label_extraction.py --path_to_file data.xlsx --column_name report --higher_model "PathologyEmoryPubMedBERT" --all_label_model "single_tfidf" --save_predictions predictions.xlsx --save_json output.json
```


## Using Docker

Coming soon


## Annotation Tool 

[A minimal annotation tool is provided](https://github.com/thiagosantos1/HCSBC/tree/main/app/src/annotation_tool)


![Alt text](https://github.com/thiagosantos1/HCSBC/blob/main/app/imgs/annotation_tool.png?raw=true "Title")


Annotation Tool Repository : [GitHub Link](https://github.com/thiagosantos1/HCSBC/tree/main/app/src/annotation_tool)


## Contributors

Ms. Thiago Santos

Dr. Imon Banerjee

Dr. Hari Trivedi

Dr. Judy Wawira
