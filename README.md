# Risk Assessment Severity Classification

This is an open-source project that evaluates the use of Deep learning to classify the severity of a hazardous scenario and support the creation of risk assessments.

We present a chatbot that uses Rasa NLU to predict the intent of the security or safety engineer, as well as extract entities related the hazard (and request for missing ones).
The chatbot triggers a BERT model that classifies the severity of the detailed scenario and recommends it to the engineer.


### Install dependencies
```shell
# (optional) create a virtual environment
python3.8 -m venv dl4nlp
source dl4nlp/bin/activate

python3.8 -m pip install -r requirements.txt

# To deactivate the virtual environment
deactivate
```

### Evaluate the models
The [BERT model](bert.ipynb) and [Logistic regression](logistic.ipynb) can be evaluated in their Jupyter Notebooks 


### Run the chatbot
```shell
# train the model
python3.8 -m rasa train

# run actions to trigger the classifier
python3.8 -m rasa run actions

# start the interactive interface
python3.8 -m rasa shell
```
