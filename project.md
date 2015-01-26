
```r
opts_knit$set(root.dir = "../pml")
```


Practical Machine Learning -- Project
========================================================
### Jan 25, 2015
*************************

# Introduction
This project aims to explore a variety of the machine learning algorithms presented in this course by using them to  predict how "well" the exercise consisting of raising and lowering a dumbbell is performed by a set of test individuals, using information obtained from a variety of sensors. Predictions are made using the `caret` package from R.

The exercise experiment was described by Velloso et al. (2013) and their data was used for this study (available at [http://groupware.les.inf.puc-rio.br/har][datalink].

# The Weightlifting Exercise
The purpose of the weightlifting exercise (WLE) experiment is to explore three aspects of human activity recognition: specifying what a correct execution consists of, detection and correction of mistakes, and providing feedback to the user regarding the quality of exection. 

Velloso et al. used an on-body sensor approach with accelerometers, gyroscopes, and magnetometers attached to the participants' midsection, forearm, arm, and to the dumbbell itself. The authors  describe the experiment as follows. 

>Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

The correctness of the excercise was determined by a professional weightlifter.

# Materials and Methods
The purpose of this project is to determine, based on the various on-body sensor data, which category (A, B, C, E, or E) any given set of measurements falls into. 

First, after correctly setting the working directory, the raw dataset was read into R

```r
library(Hmisc)
```

```
## Loading required package: grid
## Loading required package: lattice
## Loading required package: survival
## Loading required package: splines
## Loading required package: Formula
## 
## Attaching package: 'Hmisc'
## 
## The following object(s) are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
alldata_raw = csv.get("pml-training.csv")
```

```
## 160 variables; Processing variable:1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160
```

giving a dataset of 160 variables. The data was then cleaned to remove all columns containing `na` or empty values.

```r
alldata = alldata_raw[, colSums(is.na(alldata_raw)) == 0]
alldata = alldata[, colSums(alldata == "") == 0]
```

giving a dataset of 60 variables, the first seven of which are identification variables and are not used for prediction.

For cross-validation, the training data was partitioned into a training (70%) and testing (30%) dataset

```r
library(caret)
```

```
## Loading required package: ggplot2
## 
## Attaching package: 'caret'
## 
## The following object(s) are masked from 'package:survival':
## 
##     cluster
```

```r
idxtrain = createDataPartition(alldata$classe, p = 0.7, list = FALSE)
datatrain = alldata[idxtrain, ]
datatest = alldata[-idxtrain, ]
```


In order to determine potential candidate predictors, an exploratory analysis using `caret`'s `featurePlot` function with plot type "pairs" was carried out. Based on these plots, combined with an educated guess about which parameters might affect predictions for each class, the following set of variables was identified
```
'total.accel.belt',
'accel.forearm.y','accel.forearm.z',
'gyros.dumbbell.y','gyros.dumbbell.z',       
'gyros.forearm.y','gyros.forearm.z',
'gyros.arm.y','gyros.arm.z',
'magnet.dumbbell.x','magnet.dumbbell.y','magnet.dumbbell.z',
'roll.dumbbell',
'pitch.dumbbell','pitch.arm'
```

The `train` command from `caret` was used to carry out predictions using three different algorithms: classification trees using `rpart` and the above parameter set, principal compontent analysis (PCA) using `rpart` and the complete parameter set, random forests using `rf` with the complete parameter set.

# Results
The classification using `rpart` with the raw and PCA-transformed data yielded generally poor results. The classification accuracy of the former was approximately 52% and the latter 36%. Therefore, the focus in the remaining discussion is on the random forest method.

First, from the cleaned data, exclude the non-predictor columns

```r
cols = c("X", "user.name", "raw.timestamp.part.1", "raw.timestamp.part.2", "cvtd.timestamp", 
    "new.window", "num.window")
datatrain_subset = datatrain[, !names(datatrain) %in% cols]
```

Now, train the model and fit the training data to obtain an in-sample estimate of the error. The nodesize, sample size, and number of trees of the `rf` method were tweaked to decrease the time required to train the model. 

```r
modFit = train(classe ~ ., method = "rf", data = datatrain_subset, nodesize = 5, 
    do.trace = F, sampsize = 100, ntree = 150)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object(s) are masked from 'package:Hmisc':
## 
##     combine
## 
## 
## Attaching package: 'e1071'
## 
## The following object(s) are masked from 'package:Hmisc':
## 
##     impute
```

```r
pred1 = predict(modFit, newdata = datatrain)
confusionMatrix(pred1, datatrain$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3526  438  240  179   58
##          B   43 1783  143   36  285
##          C   58  318 1935  255  205
##          D  203   85   68 1629  134
##          E   76   34   10  153 1843
## 
## Overall Statistics
##                                         
##                Accuracy : 0.78          
##                  95% CI : (0.773, 0.787)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.721         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.903    0.671    0.808    0.723    0.730
## Specificity             0.907    0.954    0.926    0.957    0.976
## Pos Pred Value          0.794    0.779    0.698    0.769    0.871
## Neg Pred Value          0.959    0.924    0.958    0.946    0.941
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.257    0.130    0.141    0.119    0.134
## Detection Prevalence    0.323    0.167    0.202    0.154    0.154
## Balanced Accuracy       0.905    0.813    0.867    0.840    0.853
```

The in-sample accuracy is satisfactory (between 77% and 80% as observed from several runs).

If we apply the trained model to our testing data, we obtain an estimate of the out-of-sample error

```r
pred2 = predict(modFit, newdata = datatest)
confusionMatrix(pred2, datatest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1483  176   98   77   36
##          B   14  761   67   11  101
##          C   23  150  824  148   86
##          D  111   42   29  651   55
##          E   43   10    8   77  804
## 
## Overall Statistics
##                                         
##                Accuracy : 0.769         
##                  95% CI : (0.758, 0.779)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.706         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.886    0.668    0.803    0.675    0.743
## Specificity             0.908    0.959    0.916    0.952    0.971
## Pos Pred Value          0.793    0.798    0.669    0.733    0.854
## Neg Pred Value          0.952    0.923    0.957    0.937    0.944
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.252    0.129    0.140    0.111    0.137
## Detection Prevalence    0.318    0.162    0.209    0.151    0.160
## Balanced Accuracy       0.897    0.814    0.860    0.814    0.857
```

The accuracy, along with the sensitivities and specificities are similar for the testing set, indicating a good model fit.

# Conclusion
The random forest method proved to be the most effective, giving an estimated out-of-sample accuracy between 77% and 80%. The disadvantage compared to the other methods trialled is the increased running time.

## References
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

[datalink]: http://groupware.les.inf.puc-rio.br/har
