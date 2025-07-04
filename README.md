<div align="center">
    <img src="docs/pipegenie_logo.png" alt="PipeGenie" height="200px"/>
</div>

# PipeGenie

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.10, 3.11, 3.12-blue">
    </a>
    <a href="https://doi.org/10.1016/j.asoc.2024.111292">
        <img alt="DOI" src="https://img.shields.io/badge/DOI-10.1016%2Fj.asoc.2024.111292-blue">
    </a>
</div>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
    - [Requirements](#requirements)
    - [Installing the Package](#installing-the-package)
- [Usage](#usage)
    - [Classification Example](#classification-example)
    - [Regression Example](#regression-example)
    - [Saving and Loading Models](#saving-and-loading-models)
- [PipeGenie's Internals](#pipegenies-internals)
- [Tutorials](tutorials)
- [FAQ](#faq)
- [Contributors](#contributors)
- [Citing PipeGenie](#citing-pipegenie)

## Introduction

PipeGenie is a powerful Python tool for automating the generation and optimization of machine learning pipelines. By leveraging grammar-guided genetic programming (G3P), PipeGenie enables both novices and experts to efficiently create robust machine learning workflows. The intuitive design ensures compatibility with the widely-used scikit-learn library and its pipeline API, making it accessible for a broad audience.

Whether you're a data scientist looking to streamline your workflow or a developer exploring machine learning, PipeGenie simplifies the process of pipeline optimization, allowing you to focus on achieving the best results. With support for both classification and regression tasks, PipeGenie is a versatile tool that can help you tackle a wide range of machine learning challenges.

## Features

- **Automated Pipeline Generation**: Simplifies the creation of machine learning pipelines.
- **Grammar-Guided Genetic Programming**: Uses a set of predefined rules to evolve pipelines efficiently.
- **Compatibility**: Seamlessly integrates with scikit-learn.
- **User-Friendly**: Minimal machine learning knowledge required to get started.
- **Support for Classification and Regression**: Capable of handling both types of tasks.
- **Future Plans**: Support for time series.

## Installation

### Requirements
- **Windows:** Python **≥ 3.10, < 3.13**
- **macOS/Linux:** Python **≥ 3.10, < 3.13**
- `pip` and `setuptools` should be up to date:
  
```bash
python -m pip install --upgrade pip setuptools
```

### Installing the Package

Before installing `PipeGenie`, it is recommended to create a virtual environment to avoid conflicts with other packages. To create a virtual environment, you can use the following command:

```bash
python -m venv pipegenie-env
```

After creating the virtual environment, you can activate it using the following command:

```bash
source pipegenie-env/bin/activate
```

You can install `PipeGenie` using `pip`:

```bash
pip install pipegenie
```

You can also install `PipeGenie` from source by cloning the repository, navigating to the root directory of the repository, and running:

```bash
pip install .
```

or installing it in editable mode:

```bash
pip install -e .
```

If you see a Python version error, ensure your system meets the minimum requirements.

## Usage

**NOTE**: for Windows compatibility, you need to include `multiprocessing.freeze_support()` within a conditional block that checks if the script is being run as the main program, and specifically call it at the very beginning of that block. However, PipeGenie uses multiprocessing to speed up the evolutionary process and an evaluation timeout to avoid long-running pipelines. When the evaluation timeout is reached, the process is terminated, and the pipeline is considered invalid. While completing the task and generating all related files, Windows has some issues terminating this process, raising warnings and errors every time it tries to do so.

```python
...
from sklearn.datasets import load_iris
import multiprocessing # Import the multiprocessing module

if __name__ == '__main__':
    # Call freeze_support() only if running on Windows
    multiprocessing.freeze_support()

    # Load the dataset
    X, y = load_iris(return_X_y=True)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    ...
```

### Classification Example

You can try the following code to generate a model for a classification task using the `PipegenieClassifier` class:

```python
from pipegenie.classification import PipegenieClassifier
from pipegenie.model_selection import train_test_split
from pipegenie.metrics import balanced_accuracy_score
from sklearn.datasets import load_iris

# Load the dataset
X, y = load_iris(return_X_y=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the grammar file
grammar_file = "tutorials/sample-grammar-classification.xml"

# Create an instance of the PipegenieClassifier
model = PipegenieClassifier(
    grammar=grammar_file,
    generations=5,
    pop_size=50,
    elite_size=5,
    n_jobs=5,
    seed=42,
    outdir="sample-results",
)

# Fit the classifier
model.fit(X_train, y_train)

# Predict using the classifier
y_pred = model.predict(X_test)

# Evaluate the classifier using the default scoring method
print(f"Model score: {model.score(X_test, y_test)}")

# Evaluate the classifier using another metric
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred)}")
```

You can also check the following Jupyter notebook for a more detailed example: [Classification Example](tutorials/classification_example.ipynb).

### Regression Example

You can try the following code to generate a model for a regression task using the `PipegenieRegressor` class:

```python
from pipegenie.regression import PipegenieRegressor
from pipegenie.model_selection import train_test_split
from pipegenie.metrics import root_mean_squared_error
from sklearn.datasets import load_diabetes

# Load the dataset
X, y = load_diabetes(return_X_y=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)

# Define the grammar file
grammar_file = "tutorials/sample-grammar-regression.xml"

# Create an instance of the PipegenieRegressor
model = PipegenieRegressor(
    grammar=grammar_file,
    generations=5,
    pop_size=50,
    elite_size=5,
    maximization=False,
    n_jobs=5,
    seed=9,
    outdir="sample-results",
)

# Fit the regressor
model.fit(X_train, y_train)

# Predict using the regressor
y_pred = model.predict(X_test)

# Evaluate the regressor using the default scoring method
print(f"Model score: {model.score(X_test, y_test)}")

# Evaluate the regressor using another metric
print(f"Root mean squared error: {root_mean_squared_error(y_test, y_pred)}")
```

You can also check the following Jupyter notebook for a more detailed example: [Regression Example](tutorials/regression_example.ipynb).

### Saving and Loading Models

Once you have trained a model, you may want to save it to disk for later use. To save a model, you can use the provided `save_model` method:

```python
model.save_model("model.pkl")
```

This method will save the internal ensemble created after the evolutionary process.

In case you want to save the model manually, you can use the following code:

```python
import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(model.ensemble, file)
```

To load a model, you can use the `pickle` module:

```python
import pickle

with open("model.pkl", "rb") as file:
    ensemble = pickle.load(file)
```

## PipeGenie's Internals

### Grammar Definition

To be able to use `PipeGenie`, first you need to define the grammar that will be used to generate the pipelines. The grammar is a XML file that contains the rules that will be used to generate the pipelines. The grammar file should be structured as follows:

- **\<grammar> \</grammar>** : The root element of the file. All components of the grammar must be within these tags.
- **\<root-symbol> \</root-symbol>** : The initial symbol of the grammar. Derivations start from this symbol to create an algorithm flow and its corresponding hyperparameters. Should be written within the `<grammar>` tags.
- **\<non-terminals> \</non-terminals>** : This tag contains all the production rules tags that can be used by the evolutionary process to generate a pipeline. Should be written within the `<grammar>` tags.
- **\<non-terminal> \</non-terminal>** : Specifies a context-free grammar non terminal symbol. Should be written within the `<non-terminals>` tags. Its attribute is:
    - **name** : Name of the antecedent of the production rule.
- **\<production-rule> \</production-rule>** : Specifies the consequent of a production rule. Terminals and non-terminals can be used in the production rule and should be separated by a `;`. The symbols of the consequent have to be written with the same name as the `<terminal>` and `<non-terminal>` tags. To specify that an algorithm has hyperparameters, either terminals (list of hyperparameter `<terminal>` tags separated by a `;`) or a non-terminal (name of the non-terminal tag that contains the hyperparameters) should be used and written between parentheses (e.g., `RandomForestClassifier(rf::n_estimators;rf::max_depth)` or `RandomForestClassifier(rf_hyperparameters)`). Should be written within the `<non-terminal>` tags.
- **\<terminals> \</terminals>** : This tag contains both algorithms and hyperparameters tags that can be used by the evolutionary process to generate a pipeline. Should be written within the `<grammar>` tags.
- **\<terminal> \</terminal>** : Used to specify a terminal symbol of the grammar. It can be an algorithm or a hyperparameter. Should be written within the `<terminals>` tags. To define an algorithm, the attributes are:
    - **name** : Identifier name for the algorithm for string representation.
    - **type** : Category of the algorithm for grouping purposes. This string can contain any value, allowing grouping of terminals with the same string value. Optional.
    - **code** : Package and name of the algorithm to be imported to use in the evolutionary process (e.g., `sklearn.ensemble.RandomForestClassifier`). It can be any algorithm compatible with scikit-learn API. Otherwise, a custom wrapper should be created to adapt the API.

    To define a hyperparameter, the attributes are:
    - **name** : Identifier name for the hyperparameter. Should be written as `algorithm_id::hyperparameter_name` (e.g., `rf::n_estimators`) and should be unique for each hyperparameter. `algorithm_id` can be any string that identifies the algorithm to which the hyperparameter belongs. It is used in order to avoid conflicts between hyperparameters with the same name in different algorithms. `hyperparameter_name` should be the same as the hyperparameter name in the algorithm's constructor.
    - **type** : Data type of the hyperparameter (int, float, categorical, bool, estimator, tuple or union). For a fix value, use the prefix `fix_`.
    - **lower** : Lower limit for numerical hyperparameters.
    - **upper** : Upper limit for numerical hyperparameters.
    - **log** : Specifies whether the hyperparameter should be sampled in log scale. Only used for numerical hyperparameters.
    - **step** : Specifies the step when sampling the hyperparameter. Only used for numerical hyperparameters.
    - **values** : Specifies the possible values for categorical hyperparameters. Only used for categorical hyperparameters.
    - **estimators** : Points to a `<non-terminal>` tag that contains the possible algorithms that can be used as hyperparameters. Only used for estimator hyperparameters.
    - **value** : Specifies the value for fixed hyperparameters. Only used for fixed hyperparameters.
    - **default** : Default value for the hyperparameter.
- **\<tuple> \</tuple>** : Specifies a tuple of values for a hyperparameter (e.g., `MLPClassifier::hidden_layer_sizes`). Should be written within a `<terminal>` tag whose type is `tuple`. Its attributes are the same as the `<terminal>` tag to define hyperparameters, except for the `name` attribute.
- **\<union> \</union>** : Specifies a union of value types that can be used for a hyperparameter (e.g. `RandomForestClassifier::max_features` can be either a categorical, integer or float hyperparameter). Should be written within a `<terminal>` tag whose type is `union`. Its attributes are the same as the `<terminal>` tag to define hyperparameters, except for the `name` attribute.

A full example of grammar files can be found in the `tutorials` folder.

- [Classification Grammar](tutorials/sample-grammar-classification.xml)
- [Regression Grammar](tutorials/sample-grammar-regression.xml)

To modify the grammar, you can create your own grammar file or modify the existing ones. An explanation of how to change the grammar file can be found [here](tutorials/change-grammar.md).

### Outputs

After running the code, you will find a folder named `sample-results` (or the name you specified in the `outdir` parameter) containing the following files:

- config.txt: Contains the configuration (parameters passed to the constructor) used for the evolutionary process. An example of this file can be found [here](tutorials/sample-results/config.txt).
- best_pipeline.txt: Contains the best pipeline found by the evolutionary process, its fitness value and its predictions. An example of this file can be found [here](tutorials/sample-results/best_pipeline.txt).
- ensemble.txt: Contains the ensemble of pipelines found by the evolutionary process, their individual fitness values, the global fitness value, the weights of the pipelines in the ensemble and the predictions of the ensemble. An example of this file can be found [here](tutorials/sample-results/ensemble.txt).
- evolution.txt: Contains the evolution of the population during the evolutionary process. For each generation, it shows the number of evaluated individuals, the minimum, maximum, average and standard deviation of the fitness values of the population, fitness values of the elite, pipeline sizes of the population and pipeline sizes of the elite. At the end of the process, it will show the time in seconds that the process took. An example of this file can be found [here](tutorials/sample-results/evolution.txt).
- individuals.tsv: Contains all the individuals evaluated during the evolutionary process. For each individual, it shows the pipeline, its fitness value and the time it took to evaluate it. If the individual was invalid, it will show the reason. An example of this file can be found [here](tutorials/sample-results/individuals.tsv).

For more information about the outputs, you can check the [Understanding PipeGenie Outputs](tutorials/understanding-outputs.md) tutorial.

## FAQ

**Q: What is grammar-guided genetic programming (G3P)?**

A: G3P is a form of genetic programming that uses a grammar to guide the evolution of programs, ensuring they adhere to predefined structures and rules.

**Q: Can I use PipeGenie with other machine learning libraries?**

A: PipeGenie is designed to work with scikit-learn, so any library that is API-compatible with scikit-learn should work. If you want to use PipeGenie with a different library, you may need to make wrapper classes to adapt the API.

**Q: Why the `save_model` method only saves the ensemble and not the Pipegenie object?**

A: The ensemble is the most important part of the Pipegenie object, as it is responsible for making predictions. The Pipegenie object is only used to create and train the ensemble, so it is not necessary to save it. Also, due to limitations in the `pickle` module, it is not possible to save the Pipegenie object. In case you want to save the Pipegenie object (save the ensemble as well as the loaded grammar and configuration), you should use other serialization libraries like `dill` or `cloudpickle`.

## Contributors

- Rafael Barbudo Lunar
- Mario Berrios Carmona
- Carlos García Martínez
- Vincent Gardies
- Ángel Fuentes Almoguera
- Aurora Ramírez Quesada

- [José Raúl Romero Salguero](mailto:jrromero@uco.es)

## Citing PipeGenie

If you are using PipeGenie in your research or project, please cite using the following BibTeX entry:

```bibtex
@article{BARBUDO2024111292,
    title   = {Grammar-based evolutionary approach for automated workflow composition with domain-specific operators and ensemble diversity},
    journal = {Applied Soft Computing},
    volume  = {153},
    pages   = {111292},
    year    = {2024},
    issn    = {1568-4946},
    doi     = {https://doi.org/10.1016/j.asoc.2024.111292},
    url     = {https://www.sciencedirect.com/science/article/pii/S1568494624000668},
    author  = {Rafael Barbudo and Aurora Ramírez and José Raúl Romero},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
