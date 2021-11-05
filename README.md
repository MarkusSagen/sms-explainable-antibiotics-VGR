
# Overview
This repo is based on the work done by Omar Contreras during his MSc thesis for Peltarion, working on data from Folktandvården Västra Götaland (FTV) during the [SMS project](https://www.ai.se/en/events/pioneering-natural-language-processing-nlp-swedish-healthcare). The goal of the project was to identify based on patient journal notes for dental visits if antibiotics should be perscribed; and if so why using explainable AI methods (XAI). The model should learn to identify in which cases antibiotics were correctly or incorrecly perscribed and help mitigate the number of incorrect antibiotics presriptions    

For those interested, Omar's thesis can be found [here](https://www.diva-portal.org/smash/get/diva2:1605539/FULLTEXT01.pdf)  

# Setup   
```shell
git clone git@github.com:Peltarion/ai_explainable_antibiotics_omars_thesis.git
cd ai_explainable_antibiotics_omars_thesis
```
```shell
# Activate the virtual environment of your choise
virtualenv xai
source venv/bin/activate
pip install -r requirements.txt
```

# Running the code
Requires `jupyter notebook/lab`
```shell
cd notebooks
jupyter notebook
```
   
**NOTE**:    
For GDPR reasons, the actual datasets or trained models are not included.    
Therefore, replace in the notebooks and script files:   
- `../models/<MODEL_NAME>`  and
- `../data/<DATASET.csv>`  


# Train your own models
Recommend you convert the notebook 03_top_models.ipynb to a script or use on of the existing scripts in the [script](./scripts) folder. You will need to change the path to the data to your own data in CSV format and include a Callback for experiment logging and we advise using [Neptune](https://neptune.ai/) or [W&B](https://wandb.ai/site) for experiment logging and tracking, and is added as a [Callbacks](https://huggingface.co/transformers/main_classes/callback.html) argument to the traniner(). For an example of using an experiment logging callback, see [notebooks/03_Top_models.ipynb](./notebooks/03_Top_models.ipynb) -> Training -> callbacks - then insert your own callback.

# Inspect previously trained models 
Requires a browser and Tensorboard (included when installing requirements.txt)  
```shell
tensorboard --logdir logs
```
Then open [http://localhost:6006](http://localhost:6006)   


# Explaining the Repo
The code for the repo is structured into notebooks in the [notebooks](./notebooks) folder.  
Notebooks are named and ordered sequentially, explaining the thought process in each stage.   
The leading numbers in the names indicate the order in which they were ran.  

### Names in the notebook and meaning

<details>
<summary>00_ </summary>
Cleaning the original dataset (Excel file) and store into csv files.  
Also a notebook for identifying named entities (NER), some of which were used to clean the dataset further and annonymize the data.   
</details>
  
<details>
<summary>01_ </summary>  
First initial model trained on the uncleaned dataset  
</details>

<details>
<summary>02_ </summary> 
Notebooks for training the different [Models](#Models) on the cleaned dataset   
</details>

<details>
<summary>03_ </summary> 
Similar to `01` and `02`, but gives an overall best setting for training the models to this problem
</details>

<details>
<summary>04_ </summary>  
Explainable AI (XAI) notebooks using either Integrated Gradients (IG) or Kernel SHAP on different datasets:

  - IMDB (toy classification dataset for testing)
  - Antibiotics (FTV target dataset)
</details>


<details>
<summary>x_ </summary>
General and simplified notebooks for showcasing some of the things described in the other notebooks

  - Toy example for training a [BERT model](./notebooks/x_train_BERT_imdb.ipynb) 
  - Toy example for explaining a pre-trained model using [Kernel SHAP](./notebooks/x_XAI_Kernel_Shap.ipynb)
</details>

## Models
Models are trained for classifying when antibiotics was correctly prescribed.
Multiple Transformer based models were tested, both models 

#### Swedish models only
KB-BERT, KB-ALBERTA, KB-ELECTRA are all pre-trained Swedish models released by Royal library of Sweden (KB)  

#### Multilingual models 
- mBERT
Multilingual variant of BERT
- XLM-R
Multilingual variant of RoBERTa, which is an improved version of BERT
- mT5
Multilingal version of T5. T5 and mT5 are text-to-text or seq-to-seq models, meaning that when training the models, the input, target and output are all in the form of free text. These are large, generative models and are interesting for several reasons

## Datasets

#### Antibiotics dataset from FTV
Dataset about journal notes from dentists in Swedish. Notes contain information on a patients state, the operations made and if antobiotics was prescribed. This is the target data and models were first trained on partially cleaned data and then on a annonymized and cleaned dataset.

#### IMDB review dataset
English sentiment (binary classification) analysis dataset for movie reviews.  
Commonly used dataset for testing classification models and for toy examples.

## XAI
Explainable AI (XAI) is a field conserned with explaining a models predition.
Since Deep learning models, including Transformer based models are seen as black-box models, methods for explaining those predictions are a vital step for ensuring trust in the models predictions and can also help in detecting errors in the trained model and dataset. For instance, using Kernel SHAP on pre-trained models, we could identify that models used the localtion of the cliinc when predicting antibiotics prescription and not the journal text.  

In this notebook, we have focused on two XAI methods which are used ad-hoc to pre-trained models:
- Kernel SHAP
- Integrated Gradients (IG) 



# FAQ

1. Where can I find the code for the different models?  
   See the [notebooks](./notebooks) folder
2. If I want an easy example illustrating how to BERT for classification, where can that be found?   
   See here [notebooks/x_train_BERT_imdb.ipynb](./notebooks/x_train_BERT_imdb.ipynb)   
3. If I want an easy example illustrating how to use XAI on a pre-trained model, where can that be found?   
   See here [notebooks/x_XAI_Kernel_Shap.ipynb](./notebooks/x_XAI_Kernel_Shap.ipynb)   

