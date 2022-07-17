![Banner](https://images.unsplash.com/photo-1565371767810-ef913a6c8315?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1920&h=400&q=40)

# Bank Note Analysis
### Classify whether given Banknotes are genuine

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/navendu-pottekkat/awesome-readme)
![GitHub issues](https://img.shields.io/github/issues-raw/navendu-pottekkat/awesome-readme)
![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)

The aim is to predict whether a given banknote is authentic or not, by using number of measures taken from a photograph.

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.



> **Features are classified into five different classes:**
1. *variance* - variance of Wavelet Transformed image (continuous)
2. *skewness* - skewness of Wavelet Transformed image (continuous)
3. *curtosis* - curtosis of Wavelet Transformed image (continuous)
4. *entropy* - entropy of image (continuous)
5. *class* - class (0 or 1)


Sometimes when someone deposite a sum of large money into bank, it is very time consuming for bank staffs to check each note authentication. In such case machine learning algorithms come to play. They simply put the money into a machine which according to some machine learning model, predicts wheather the note is authentic or not. There are many machine learning models used for such classifications, but here we focus on only three out of them.

> **Implemented Algorithms:**
1. Logistic Regression
2. Support Vector Machine (SVM)
3. kNeighbors classification


> **Problem Statement:**
To identify whether a banknote is real or not, we needed a dataset of real as well as fake banknotes along with their different features.

We know that data is messy. A dataset may contain multiple missing values. In that situation, we have to clean the dataset. To avoid this kind of hassle we are going to use a pre-cleaned dataset. You can download the dataset (.CSV file) from [here](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

``` pd.read_csv('data.csv') ``` - convert to pandas dataframe

``` data.isna().sum() ``` - check wheather this dataset contains any empty/null value or

> **Data visualization:**

![dataset](https://github.com/Abhishek-k-git/BankNote-Analysis/blob/main/images/data_head.png)

After dataprocessing or cleaning, it is very crucial to visualize dataset, there are many datavisualization tool out there. But here we use [seaborn](https://seaborn.pydata.org/), which is a python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

> **Training / Testing data split:**

Now data is divided into two sets one is *training dataset* which is used to train the model (just like a new born child learns by sensing things around him), the other dataset is *testing dataset* which is used to evaluate or predict the accuracy of model. The machine uses its model, apply to testing dataset to give out predicted results. The predicted output then compared to final result in actual dataset (In this case it is labeled as *class*). That's why it is necessary to first drop that column named class, before we train our model.

```
X = data.drop('class', axis = 1)
y = data['class']

# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)
```
> **Data preprocessing**
Scale the independent variables in dataset to normalize the data within a particular range. It also helps in speeding up the calculations in an algorithm. Most of the times, collected data set contains features highly varying in magnitudes, units and range.

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
```

> **Data Modelling**
#### 1. Logistic Regression
Logistic regression estimates the probability of an event occurring, such as occured or didn't occured, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1
```
 Accuracies :
 [0.98181818 0.99090909 0.98181818 0.99090909 0.99090909 0.99090909
 0.96363636 0.99082569 0.97247706 0.98165138]
 
 Mean Accuracy :  0.9835863219349459
 ```

![matrix](https://github.com/Abhishek-k-git/BankNote-Analysis/blob/main/images/matrix_logisticReg.png)

#### 2. Support Vector Machine (SVM)
It is used to handle both classification and regression on linear and non-linear data. SVMs are used in applications like handwriting recognition, intrusion detection, face detection, email classification, gene classification, and in web pages.
```
 Accuracies :
 [0.99090909 0.99090909 0.99090909 0.99090909 0.99090909 0.99090909
 0.96363636 0.99082569 0.97247706 0.98165138]
 
 Mean Accuracy :  0.9854045037531277
 ```

![matrix](https://github.com/Abhishek-k-git/BankNote-Analysis/blob/main/images/matrix_svm.png)


#### 2. kNeighbors classification
The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.
```
 Accuracies :
 [0.99090909 0.99090909 0.99090909 0.99090909 0.99090909 0.99090909
 0.96363636 0.99082569 0.97247706 0.98165138]

 Mean Accuracy :  0.9854045037531277
 ```

![matrix](https://github.com/Abhishek-k-git/BankNote-Analysis/blob/main/images/matrix_kneigh.png)

