# Understanding PipeGenie Outputs

As mentioned in the [Outputs](../README.md#outputs) section, PipeGenie generates some files that we will explain in this tutorial.

## Configuration File

The configuration file ([config.txt](sample-results/config.txt)) is a text file that contains all the parameters used for the evolutionary process, so they can be used to reproduce the results. An example of a configuration file is shown below:

```txt
grammar: sample-grammar-classification.xml
grammar_type: evoflow-xml
pop_size: 50
generations: 5
fitness: balanced_accuracy_score
nderiv: 13
selection: TournamentSelection(tournament_size=3)
crossover: MultiCrossover(p_crossover=0.8)
mutation: MultiMutation(p_mutation=0.5, p_hyper=0.5)
mutation_elite: HyperparameterMutation(p_mutation=0.2)
replacement: ElitistGenerationalReplacement(elitism=5)
use_double_mutation: False
elite_size: 5
timeout: 3600
eval_timeout: 360
cross_validator: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
maximization: True
early_stopping_threshold: 0.0001
early_stopping_generations: None
early_stopping_time: None
outdir: sample-results
n_jobs: 5
seed: 42
use_predict_proba: False
diversity_weight: 0.2
```

From the configuration file, we can see the following parameters:

- **grammar**: the loaded grammar file where the pipeline structure and rules to generate the pipelines are defined. In this case, the grammar file was [sample-grammar-classification.xml](sample-grammar-classification.xml).
- **grammar_type**: the type of grammar file used. In this case, the grammar file was in the evoflow-xml format.
- **pop_size**: the population size used in the evolutionary process. In this case, the population consisted of 50 individuals.
- **generations**: maximum number of generations used in the evolutionary process, excluding the initial generation where the population is randomly generated. In this case, the evolutionary process was run for up to 5 generations plus the initial generation.
- **fitness**: the fitness function used to assess the performance of the pipelines. In this case, as the task was a classification problem, the balanced accuracy score was used.
- **nderiv**: the maximum number of derivations used to generate a syntax tree. In this case, the maximum number of derivations was 13.
- **selection**: the selection operator used to select the individuals that will be used to generate the offspring. In this case, the tournament selection operator with a tournament size of 3 was used, meaning that three individuals are randomly selected from the population, and the one with the best fitness value is chosen.
- **crossover**: the crossover operator used to generate the offspring. In this case, the multi-crossover operator was employed with a crossover probability of 80%. This operator checks the common hyper-parameters between the individuals. If the individuals have more than one common hyper-parameter, the crossover is performed on the hyper-parameters. Otherwise, the crossover is performed on the structure of the individuals.
- **mutation**: the mutation operator used to mutate the individuals. In this case, the multi-mutation operator was employed with a mutation probability of 50%. This operator, randomly, selects between mutating hyper-parameters, which occurs with a probability of 50%, or mutating the structure of the individual.
- **mutation_elite**: the mutation operator used to mutate the elite individuals. In this case, the hyperparameter mutation operator, which mutates the hyper-parameters of the individual with a inversely proportional probability to the number of hyper-parameters, was employed with a mutation probability of 20%.
- **replacement**: the replacement operator used to replace the individuals in the population. In this case, the elitist generational replacement operator was employed with an elitism of 5. This operator replaces the whole population with the offspring. Then, the specified number of elite individuals are added to the new population, replacing the worst individuals.
- **use_double_mutation**: a boolean parameter that indicates whether to use the crossover and mutation operators to generate the offspring, or only the mutation operators. The `mutation_elite` is used to mutate the elite individuals, while the `mutation` operator is used to mutate the rest of the population. In this case, the `use_double_mutation` parameter was set to `False`, meaning that `mutation_elite` was not used.
- **elite_size**: the number of elite individuals to be preserved. In this case, the 5 best individuals were preserved.
- **timeout**: the maximum time in seconds that the evolutionary process can run. In this case, the evolutionary process was set to run for up to 3600 seconds (1 hour).
- **eval_timeout**: the maximum time in seconds that the evaluation of a pipeline can run. In this case, the evaluation of a pipeline was set to run for up to 360 seconds (6 minutes).
- **cross_validator**: the cross-validation strategy used to evaluate the pipelines. In this case, the stratified k-fold cross-validation strategy with 5 splits was used.
- **maximization**: a boolean parameter that indicates whether the fitness function should be maximized or minimized. In this case, as the balanced accuracy score was used, the `maximization` parameter was set to `True`.
- **early_stopping_threshold**: the threshold used to determine if there is improvement in the elite average fitness between generations. In this case, the threshold was set to 0.0001.
- **early_stopping_generations**: the number of generations without improvement in the elite average fitness that will trigger the early stopping mechanism. In this case, the early stopping mechanism was not used.
- **early_stopping_time**: the maximum time in seconds that the evolutionary process can run without improvement in the elite average fitness that will trigger the early stopping mechanism. In this case, the early stopping mechanism was not used.
- **outdir**: the output directory where the results will be saved. In this case, the results were saved in the `sample-results` directory.
- **n_jobs**: the number of jobs to run in parallel. In this case, 5 jobs were run in parallel.
- **seed**: the seed used to initialize the random number generator. In this case, the seed was set to 42.
- **use_predict_proba**: a boolean parameter that indicates whether to use the `predict_proba` method to evaluate the pipelines. In this case, as the fitness function used was the balanced accuracy score which requires the class labels, the `use_predict_proba` parameter was set to `False`.
- **diversity_weight**: the weight of the prediction diversity in the individuals' fitness. In this case, the weight was set to 0.2.

## Best Pipeline

The best pipeline file ([best_pipeline.txt](sample-results/best_pipeline.txt)) contains the best pipeline found during the evolutionary process. An example of a best pipeline file is shown below:

```txt
fastICA(84,'deflation','unit-variance','exp');adaBoost(decisionTree('entropy',24,2,0.10264332148015243,14,'balanced'),29,0.9997595874046005,'SAMME')
Fitness: 0.9714285714285715
Prediction: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
 0 0 0 2 1 1 0 0 1 2 2 1 2 1 2 1 0 2 1 0 0 0 1 2 0 0 0 1 0 1 2 0 1 2 0 2 2
 1 1 2 1 0 1 2 0 0 1 1 0 2 0 0 1 1 2 1 2 2 1 0 0 2 2 0 0 0 1 2 0 2 2 0 1 1
 2]
```

We can check that the best pipeline found was an AdaBoost classifier. To identify which value represents each hyper-parameter, we need to check the grammar file used in the evolutionary process. From the grammar file specified in the configuration for this example ([sample-grammar-classification.xml](sample-grammar-classification.xml)), we can see the following definition of the AdaBoost classifier:

```xml
<terminal name="adaBoost" type="ensemble" code="sklearn.ensemble.AdaBoostClassifier"/>
    <terminal name="ada::estimator" type="estimator" estimators="estimators"/>
    <terminal name="ada::n_estimators" type="int" lower="10" upper="100" default="50"/>
    <terminal name="ada::learning_rate" type="float" lower="0.01" upper="2" default="0.1" log="True"/>
    <terminal name="ada::algorithm" type="categorical" values="SAMME.R;SAMME" default="SAMME.R"/>

<non-terminal name="estimators">
    <production-rule>decisionTree(decisionTree_hp)</production-rule>
</non-terminal>
```

Considering the grammar definition, we can see that the hyper-parameters of the AdaBoost classifier are:

- **estimator**: the base estimator used for the AdaBoost classifier. In this case, the terminal node responsible for choosing the base estimator is pointing to a non-terminal node called `estimators`, whose production rule is `decisionTree(decisionTree_hp)`. This means that the base estimator is a decision tree.
- **n_estimators**: the number of estimators used in the AdaBoost classifier. In this case, the number of estimators was set to 29.
- **learning_rate**: the learning rate used in the AdaBoost classifier. In this case, the learning rate was set to 0.9997595874046005.
- **algorithm**: the algorithm used in the AdaBoost classifier. In this case, the algorithm was set to SAMME.

For the fitness value, we can see that the balanced accuracy score, specified in the configuration file, achieved with cross-validation was 0.9714285714285715.

Lastly, the class labels predicted by the best pipeline for the whole training set (the one provided to the fit method) are shown.

## Ensemble

The ensemble file ([ensemble.txt](sample-results/ensemble.txt)) contains the final ensemble generated by PipeGenie. An example of an ensemble file is shown below:

```txt
0: Pipeline(
	0: FastICA(algorithm='deflation', fun='exp', n_components=84, random_state=42)
	1: AdaBoostClassifier(algorithm='SAMME',
                   estimator=DecisionTreeClassifier(class_weight='balanced',
                                                    criterion='entropy',
                                                    max_depth=24,
                                                    max_features=0.10264332148015243,
                                                    min_samples_leaf=14,
                                                    random_state=42),
                   learning_rate=0.9997595874046005, n_estimators=29,
                   random_state=42)
) -> Fitness: 0.9714285714285715

1: Pipeline(
	0: FastICA(algorithm='deflation', fun='exp', n_components=84, random_state=42)
	1: AdaBoostClassifier(algorithm='SAMME',
                   estimator=DecisionTreeClassifier(class_weight='balanced',
                                                    max_depth=17,
                                                    max_features=0.09090941217379389,
                                                    min_samples_leaf=2,
                                                    min_samples_split=17,
                                                    random_state=42),
                   learning_rate=0.9997595874046005, n_estimators=29,
                   random_state=42)
) -> Fitness: 0.9714285714285715

2: Pipeline(
	0: FastICA(algorithm='deflation', fun='exp', n_components=84, random_state=42)
	1: AdaBoostClassifier(algorithm='SAMME',
                   estimator=DecisionTreeClassifier(class_weight='balanced',
                                                    criterion='entropy',
                                                    max_depth=22,
                                                    max_features=0.29740064768156615,
                                                    min_samples_leaf=4,
                                                    min_samples_split=11,
                                                    random_state=42),
                   learning_rate=0.9997595874046005, n_estimators=29,
                   random_state=42)
) -> Fitness: 0.961904761904762

3: Pipeline(
	0: Nystroem(coef0=0.9325292278682136, degree=5, gamma=0.20700104585316448,
         kernel='cosine', n_components=80, random_state=42)
	1: AdaBoostClassifier(algorithm='SAMME',
                   estimator=DecisionTreeClassifier(criterion='entropy',
                                                    max_depth=4,
                                                    max_features=0.5309219311457678,
                                                    min_samples_leaf=14,
                                                    min_samples_split=19,
                                                    random_state=42),
                   learning_rate=0.5987735800266115, n_estimators=27,
                   random_state=42)
) -> Fitness: 0.9619047619047618

4: Pipeline(
	0: Nystroem(coef0=0.11035242316372162, degree=5, gamma=0.20700104585316448,
         kernel='cosine', n_components=88, random_state=42)
	1: AdaBoostClassifier(algorithm='SAMME',
                   estimator=DecisionTreeClassifier(criterion='entropy',
                                                    max_depth=22,
                                                    max_features=0.5309219311457678,
                                                    min_samples_leaf=14,
                                                    min_samples_split=6,
                                                    random_state=42),
                   learning_rate=1.1708564525311265, n_estimators=55,
                   random_state=42)
) -> Fitness: 0.9619047619047618

Ensemble fitness: 0.980952380952381
Weights: [1.0, 1.0, 0.9901960784313726, 0.9901960784313724, 0.9901960784313724]
Prediction: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
 0 0 0 2 1 1 0 0 1 2 2 1 2 1 2 1 0 2 1 0 0 0 1 2 0 0 0 1 0 1 2 0 1 2 0 2 2
 1 1 2 1 0 1 2 0 0 1 1 0 2 0 0 1 1 2 1 2 2 1 0 0 2 2 0 0 0 1 2 0 2 2 0 1 1
 2]
```

It can be seen that the ensemble is composed of 5 pipelines, as specified in the configuration file. For each pipeline, its structure is shown, as well as its individual fitness value. The ensemble fitness, which is calculated using the weighted voting method (the final decision is made based on the weighted sum of votes), obtained with cross-validation is also shown. The weights, relative to the fitness of each pipeline, are also displayed. Lastly, the class labels predicted by the ensemble for the whole training set (the one provided to the fit method) are shown.

## Evolution process

The evolution process file ([evolution.txt](sample-results/evolution.txt)) shows the evolution of the population and the elite during the evolutionary process. An example of an evolution process file is shown below:

```txt
                                       fitness                                            size                                        fitness_elite                                      size_elite              
                 ----------------------------------------------------    --------------------------------------    ----------------------------------------------------    --------------------------------------
gen    nevals    min           max           avg           std           min    max    avg           std           min           max           avg           std           min    max    avg           std       
0      50        0.33333       0.97143       0.88701       0.14218       1      5      2.64          1.3381        0.95238       0.97143       0.95619       0.007619      2      3      2.2           0.4       
1      45        0.33333       0.97143       0.91138       0.10502       1      5      2.7021        1.2702        0.95238       0.97143       0.9581        0.007619      1      3      2             0.63246   
2      38        0.33333       0.97143       0.92632       0.098798      1      5      2.4651        1.1684        0.95238       0.97143       0.9619        0.0060234     1      2      1.8           0.4       
3      40        0.91759       0.97143       0.9448        0.014293      1      4      2.0732        0.7454        0.9619        0.97143       0.96381       0.0038095     1      2      1.8           0.4       
4      33        0.92381       0.97143       0.94737       0.013243      1      4      2.0263        0.62773       0.9619        0.97143       0.96381       0.0038095     1      2      1.8           0.4       
5      30        0.92381       0.97143       0.95317       0.010618      1      4      2.25          0.86201       0.9619        0.97143       0.96571       0.0046657     2      2      2             0         
--- 74.90672183036804 sec ---
```

As we can see, each row represents a generation of the evolutionary process and shows the following information:

- **gen**: the generation number.
- **nevals**: the number of individuals evaluated in the generation. This number may be less than the population size if an individual remained the same as the previous generation, the pipeline was invalid, or the evaluation of the pipeline timed out.
- **fitness**: the minimum, maximum, average, and standard deviation of the fitness values of the individuals in the population.
- **size**: the minimum, maximum, average, and standard deviation of the sizes of the individual's pipelines in the population.
- **fitness_elite**: the minimum, maximum, average, and standard deviation of the fitness values of the elite individuals.
- **size_elite**: the minimum, maximum, average, and standard deviation of the sizes of the elite individuals' pipelines.

The last row shows the total time taken to run the evolutionary process.

## Individuals

The last file generated by PipeGenie is the individuals file ([individuals.tsv](sample-results/individuals.tsv)), which shows all the individuals that have been generated during the evolutionary process. An example of an individuals file is shown below:

| Pipeline | Fitness | Fit Time |
| --- | --- | --- |
| minMaxScaler();maxAbsScaler();minMaxScaler();selectPercentile(35.202865826522476);randomForest(79,'entropy','None',19,10,True,'None') | invalid_ind (duplicated operator) | invalid_ind (duplicated operator) |
| logisticRegression('None','liblinear',1000,3.0565936532910296,'balanced') | eval_error (penalty=None is not supported for the liblinear solver) | eval_error (penalty=None is not supported for the liblinear solver) |
| maxAbsScaler();varianceThreshold();pca(0.5951859161895027,False);adaBoost(decisionTree('entropy',26,5,0.24865633392028563,7,'None'),27,0.5987735800266115,'SAMME') | 0.9428571428571428 | 1.016413688659668 |
| extraTreesClassifier(53,'gini','log2',18,16,False,'balanced') | 0.9523809523809523 | 1.038949728012085 |
| selectFwe(0.03511888658454656);minMaxScaler();varianceThreshold();fastICA(29,'deflation','unit-variance','logcosh');gaussianNB(0.0004324407148895682) | 0.9238095238095239 | 1.0971026420593262 |
| fastICA(84,'deflation','unit-variance','exp');adaBoost(decisionTree('gini',17,17,0.09090941217379389,2,'balanced'),29,0.9997595874046005,'SAMME') | 0.9714285714285715 | 1.1985087394714355 |
| maxAbsScaler();minMaxScaler();mlpClassifier((30, 6),0.08765042334825297,0.01832113302637054,'adam','tanh') | 0.9333333333333333 | 1.1667122840881348 |
| fastICA(12,'parallel','unit-variance','exp');varianceThreshold();minMaxScaler();qda(0.6091310056669882) | 0.8380952380952381 | 0.9527671337127686 |
| maxAbsScaler();robustScaler(False,True);minMaxScaler();adaBoost(decisionTree('entropy',15,18,0.45148614139112975,8,'balanced'),18,0.4593793182597367,'SAMME.R') | 0.9333333333333333 | 0.9090633392333984 |
| minMaxScaler();gaussianNB(0.001893137230228278) | 0.9523809523809526 | 0.836543083190918 |
| maxAbsScaler();qda(0.8218025940068083) | 0.8476190476190476 | 0.8256945610046387 |
... (more individuals)

Each row of this file represents an individual that was generated during the evolutionary process, showing the following information:

- **pipeline**: the pipeline structure of the individual.
- **fitness**: the fitness value of the individual.
- **fit_time**: the time taken to cross-validate the individual.

For the individuals that were invalid, the fitness and fit_time columns show the reason why the individual was invalid. In this example, the first 2 individuals were invalid. The first one was invalid because it had a duplicated operator (minMaxScaler), and the second one was invalid because a parameter combination was not supported (penalty=None is not supported for the liblinear solver).