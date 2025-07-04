# How to change the grammar file?

During the usage of `PipeGenie`, you may want to change the grammar file to generate pipelines with different rules, other algorithms, or change the possible hyperparameter values. To do this, you need to create a new grammar file or modify an existing one. The grammar file should be structured as described in [Grammar Definition](../README.md#grammar-definition) section of the [README](../README.md) file.

## Adding and removing algorithms from the grammar

Consider the following example of a simple grammar file for classification tasks with a linear classifier and two preprocessing algorithms:

```xml
<grammar>
    <!--  Root -->
    <root-symbol>workflow</root-symbol>
    <!-- Terminals -->
    <terminals>
        <!-- PREPROCESSING -->
        <!-- scaler -->
        <terminal name="minMaxScaler" type="scaler" code="sklearn.preprocessing.MinMaxScaler"/>
        <!-- featureSelection -->
        <terminal name="selectPercentile" type="featureSelection" code="sklearn.feature_selection.SelectPercentile"/>
            <terminal name="sp::percentile" type="float" lower="1" upper="99" default="50"/>
        <!-- CLASSIFIERS --> 
        <!-- linear -->
        <terminal name="logisticRegression" type="linear" code="sklearn.linear_model.LogisticRegression"/>
            <terminal name="lr::penalty" type="categorical" values="l1;l2;None" default="l2"/>
            <terminal name="lr::solver" type="categorical" values="lbfgs;liblinear;newton-cg;sag;saga" default="lbfgs"/>
            <terminal name="lr::max_iter" type="fix_int" value="1000"/>
            <terminal name="lr::C" type="float" lower="1e-4" upper="25" default="1" log="True"/>
            <terminal name="lr::class_weight" type="categorical" values="balanced" default="None"/>
    </terminals>
    <!-- Non terminals: production rules -->
    <non-terminals> 
        <!-- Structure definition -->
        <non-terminal name="workflow">
            <production-rule>classifier</production-rule>
            <production-rule>preprocessingBranch;classifier</production-rule>
        </non-terminal>
        <non-terminal name="preprocessingBranch">
            <production-rule>preprocess</production-rule>
            <production-rule>preprocessingBranch;preprocess</production-rule>
        </non-terminal>
        <!-- CLASSIFIERS -->
        <non-terminal name="classifier">
            <!-- linear -->
            <production-rule>logisticRegression(logisticRegression_hp)</production-rule>
        </non-terminal>
        <!-- PREPROCESSING -->
        <non-terminal name="preprocess">
            <!-- scaler -->
            <production-rule>minMaxScaler</production-rule>
            <!-- feature selection-->
            <production-rule>selectPercentile(selectPercentile_hp)</production-rule>
        </non-terminal>
        <!-- HYPERPARAMETERS -->
        <non-terminal name="logisticRegression_hp">
            <production-rule>lr::penalty;lr::solver;lr::max_iter;lr::C;lr::class_weight</production-rule>
        </non-terminal>
        <non-terminal name="selectPercentile_hp">
            <production-rule>sp::percentile</production-rule>
        </non-terminal>
    </non-terminals>
</grammar>
```

If you wish to add a new algorithm to the grammar, for instance, a decision tree classifier, you only need to add a new terminal tag with the information of the algorithm and a terminal tag for each of the hyperparameters that you want to optimize. Then, you need to add a new production rule in the non-terminal tag that will use the new algorithm and, in case you want to optimize the hyperparameters, you need to create a new non-terminal tag with the hyperparameters and a production rule that will expand the hyperparameters of the new algorithm. The following example shows how to add a decision tree classifier to the grammar file:

```xml
<grammar>
    <!--  Root -->
    <root-symbol>workflow</root-symbol>
    <!-- Terminals -->
    <terminals>
        <!-- PREPROCESSING -->
        <!-- scaler -->
        <terminal name="minMaxScaler" type="scaler" code="sklearn.preprocessing.MinMaxScaler"/>
        <!-- featureSelection -->
        <terminal name="selectPercentile" type="featureSelection" code="sklearn.feature_selection.SelectPercentile"/>
            <terminal name="sp::percentile" type="float" lower="1" upper="99" default="50"/>
        <!-- CLASSIFIERS --> 
        <!-- linear -->
        <terminal name="logisticRegression" type="linear" code="sklearn.linear_model.LogisticRegression"/>
            <terminal name="lr::penalty" type="categorical" values="l1;l2;None" default="l2"/>
            <terminal name="lr::solver" type="categorical" values="lbfgs;liblinear;newton-cg;sag;saga" default="lbfgs"/>
            <terminal name="lr::max_iter" type="fix_int" value="1000"/>
            <terminal name="lr::C" type="float" lower="1e-4" upper="25" default="1" log="True"/>
            <terminal name="lr::class_weight" type="categorical" values="balanced" default="None"/>
        <!-- tree -->
        <terminal name="decisionTree" type="tree" code="sklearn.tree.DecisionTreeClassifier"/>   <!-- New terminal tags -->
            <terminal name="dt::criterion" type="categorical" values="gini;entropy" default="gini"/>
            <terminal name="dt::max_depth" type="int" lower="0" upper="30" default="10"/>
            <terminal name="dt::min_samples_split" type="int" lower="2" upper="20" default="2"/>
            <terminal name="dt::max_features" type="float" lower="0" upper="1" default="0.5"/>
            <terminal name="dt::min_samples_leaf" type="int" lower="1" upper="20" default="1"/>
            <terminal name="dt::class_weight" type="categorical" values="balanced" default="None"/>
    </terminals>
    <!-- Non terminals: production rules -->
    <non-terminals> 
        <!-- Structure definition -->
        <non-terminal name="workflow">
            <production-rule>classifier</production-rule>
            <production-rule>preprocessingBranch;classifier</production-rule>
        </non-terminal>
        <non-terminal name="preprocessingBranch">
            <production-rule>preprocess</production-rule>
            <production-rule>preprocessingBranch;preprocess</production-rule>
        </non-terminal>
        <!-- CLASSIFIERS -->
        <non-terminal name="classifier">
            <!-- linear -->
            <production-rule>logisticRegression(logisticRegression_hp)</production-rule>
            <!-- tree -->
            <production-rule>decisionTree(decisionTree_hp)</production-rule>                     <!-- New production rule -->
        </non-terminal>
        <!-- PREPROCESSING -->
        <non-terminal name="preprocess">
            <!-- scaler -->
            <production-rule>minMaxScaler</production-rule>
            <!-- feature selection-->
            <production-rule>selectPercentile(selectPercentile_hp)</production-rule>
        </non-terminal>
        <!-- HYPERPARAMETERS -->
        <non-terminal name="logisticRegression_hp">
            <production-rule>lr::penalty;lr::solver;lr::max_iter;lr::C;lr::class_weight</production-rule>
        </non-terminal>
        <non-terminal name="decisionTree_hp">                                                     <!-- New non-terminal tag -->
            <production-rule>dt::criterion;dt::max_depth;dt::min_samples_split;dt::max_features;dt::min_samples_leaf;dt::class_weight</production-rule>
        </non-terminal>
        <non-terminal name="selectPercentile_hp">
            <production-rule>sp::percentile</production-rule>
        </non-terminal>
    </non-terminals>
</grammar>
```

As shown in the example above, new `<terminal>` tags were added to the `<terminals>` tag pointing to the decision tree classifier algorithm, along with the hyperparameters that can be configured. To be able to use the new algorithm in the pipeline generation, a new `<production-rule>` tag was added to the `<non-terminal>` tag that contains, in this case, the classifiers. The new production rule specifies that it will expand the derivation tree adding 2 new symbols: the `decisionTree` symbol which will create the object of the DecisionTreeClassifier class and the `decisionTree_hp` symbol which will set the hyperparameters of the DecisionTreeClassifier object. The `decisionTree_hp` symbol is defined in a new `<non-terminal>` tag that contains the hyperparameters of the DecisionTreeClassifier algorithm. The production rule of the `decisionTree_hp` non-terminal specifies that it will expand the derivation tree adding 6 new symbols, each one representing a hyperparameter of the DecisionTreeClassifier algorithm.

To remove an algorithm from the grammar, you only need to remove the corresponding production rule from the non-terminal tag that contains the algorithm. This will effectively prevent the evolutionary process from generating pipelines with the algorithm, but for clarity, you may also want to remove the terminal tags that contain the algorithm information.

## Changing the pipeline structure

Another common change in the grammar file is to modify the structure of the pipeline to enforce a specific order of the algorithms and reduce the search space. For example, if your dataset has missing values, you may want to have a mandatory imputation step before any other preprocessing or classification/regression algorithm. To do this, you can add a new `<non-terminal>` containing the possible imputation algorithms and add or modify a production rule that will enforce the imputation step before any other algorithm. Another possibility, in case there is only one imputation algorithm, is to add the algorithm and the hyperparameters symbols directly to the production rule, but for clarity, you may prefer to create a new non-terminal tag for the imputation step. Keep in mind that, although it is clearer to create a new non-terminal tag, it will consume a derivation from the provided to create the derivation tree. The following example shows how to add a mandatory imputation step before the preprocessing and classification algorithms:

```xml
<grammar>
    <!--  Root -->
    <root-symbol>workflow</root-symbol>
    <!-- Terminals -->
    <terminals>
        <!-- PREPROCESSING -->
        <!-- imputation -->
        <terminal name="simpleImputer" type="imputer" code="sklearn.impute.SimpleImputer"/>      <!-- New terminal tags -->
            <terminal name="si::strategy" type="categorical" values="mean;median;most_frequent;constant" default="mean"/>
            <terminal name="si::fill_value" type="float" lower="-1e2" upper="1e2" default="0"/>
        <!-- other algorithms -->
        Preprocessing algorithms ...
        <!-- CLASSIFIERS --> 
        Classifiers ...
    </terminals>
    <!-- Non terminals: production rules -->
    <non-terminals> 
        <!-- Structure definition -->
        <non-terminal name="workflow">
            <production-rule>imputation;classifier</production-rule>                            <!-- Modified production rule -->
            <production-rule>imputation;preprocessingBranch;classifier</production-rule>        <!-- Modified production rule -->
        </non-terminal>
        <non-terminal name="preprocessingBranch">
            <production-rule>preprocess</production-rule>
            <production-rule>preprocessingBranch;preprocess</production-rule>
        </non-terminal>
        <!-- CLASSIFIERS -->
        <non-terminal name="classifier">
            Classifier production rules ...
        </non-terminal>
        <!-- PREPROCESSING -->
        <!-- imputation -->
        <non-terminal name="imputation">                                                        <!-- New non-terminal tag -->
            <production-rule>simpleImputer(simpleImputer_hp)</production-rule>
        </non-terminal>
        <!-- other preprocessing algorithms -->
        <non-terminal name="preprocess">
            Preprocessing production rules ...
        </non-terminal>
        <!-- HYPERPARAMETERS -->
        <non-terminal name="simpleImputer_hp">                                                   <!-- New non-terminal tag -->
            <production-rule>si::strategy;si::fill_value</production-rule>
        </non-terminal>
        <!-- other hyperparameters -->
    </non-terminals>
</grammar>
```

In the example above, a new `<non-terminal>` tag named `imputation` was added to the `<non-terminals>` tag containing the possible imputation algorithms. Then, the production rules of the `workflow` non-terminal were modified to include the imputation step before the possible preprocessing algorithms and the classifier.

## Changing the hyperparameters

To change the hyperparameters of an algorithm, you only need to modify the corresponding `<terminal>` tag in the `<terminals>` tag. The following example shows how to change the hyperparameters of the DecisionTreeClassifier algorithm:

```xml
<grammar>
    <!--  Root -->
    <root-symbol>workflow</root-symbol>
    <!-- Terminals -->
    <terminals>
        <!-- PREPROCESSING -->
        Preprocessing algorithms ...
        <!-- CLASSIFIERS --> 
        <!-- tree -->
        <terminal name="decisionTree" type="tree" code="sklearn.tree.DecisionTreeClassifier"/>
            <terminal name="dt::criterion" type="categorical" values="gini;entropy" default="gini"/>
            <terminal name="dt::max_depth" type="int" lower="0" upper="30" default="10"/>
            <terminal name="dt::min_samples_split" type="int" lower="2" upper="20" default="2"/>
            <terminal name="dt::max_features" type="float" lower="0" upper="1" default="0.5"/>
            <terminal name="dt::min_samples_leaf" type="int" lower="1" upper="20" default="1"/>
            <terminal name="dt::class_weight" type="categorical" values="balanced" default="None"/>
    </terminals>
    <!-- Non terminals: production rules -->
    <non-terminals> 
        <!-- Structure definition -->
        Non-terminal tags ...
    </non-terminals>
</grammar>
```

In the above example, the decision tree has been configured to have a maximum depth between 0 and 30, but you can change this range by modifying the `max_depth` hyperparameter in the `<terminal>` tag. For example, to change the maximum depth to be between 5 and 20, you can modify the `max_depth` hyperparameter as follows:

```xml
<terminal name="dt::max_depth" type="int" lower="5" upper="20" default="10"/>
```

Another possibility is to change the value type of a hyperparameter. For example, the `min_samples_split` hyperparameter of the DecisionTreeClassifier algorithm can be either an int, used in the previous example, or a float. To change the type of the `min_samples_split` hyperparameter to a float, you can modify the `<terminal>` tag as follows:

```xml
<terminal name="dt::min_samples_split" type="float" lower="0.01" upper="0.1" default="0.05"/>
```

You may also want to remove a hyperparameter from an algorithm by removing the corresponding `<terminal>` tag and modifying the production rule that expands the hyperparameters. If you remove a hyperparameter from the grammar file, the evolutionary process will not be able to set the hyperparameter value, 
so the corresponding algorithm will use the default value for the hyperparameter, or you can add a new hyperparameter by adding a new `<terminal>` to allow the evolutionary process to optimize it.

## Conclusion

In this tutorial, you learned how to change the grammar file to add or remove algorithms, change the pipeline structure, and modify the hyperparameters of the algorithms. By following these steps, you can customize the grammar file to generate pipelines that better suit your needs.

Use this [link](../README.md) to return to the main README file.