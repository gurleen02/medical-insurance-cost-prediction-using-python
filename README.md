# medical-insurance-cost-prediction-using-python

Many elements affect the expenses of medical insurance and it becomes quite tough to predict those expenses due to the variability of those elements. A regression model has been used to recognize and study a complex sample that enables us to predict the price of medical 
insurance.
Regression is a technique for  investigating the relationship between independent variables or features and a dependent variable or outcome. It’s used as a method for predictive modelling in machine learning, in which an algorithm is used to predict continuous outcomes.

## Data Source ##

    The Health Insurance dataset from Kaggle has been used and the model has been trained on attributes like age, sex, BMI, children, smoker and region. 

### Attribute table ###

S.No. | Attribute | Remarks
------|-----------|--------
1 | Age | Age of primary beneficiary
2 | Sex | Gender of patients; Female(1) and Male(0)
3 | BMI | Body mass index, providing an understanding of the body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
4 | Children | Number of children covered by medical insurance.
5 | Smoker | Smoking
6 | Region | The beneficiary, residential area in the US, North-East, South-East, South-West, and North-West.
7 | Charges | Medical Insurance price

### Flowchart of the Proposed Algorithm ###


### Step 1: Import the required modules ###

    import pandas as pd <br>
    import numpy as np <br>
    
    ----- OR -----
    
    pip install pandas
    pip install numpy
 
### Step 2: Importing the Data Set ###

    medical_datafile = pd.read_csv("Health_insurance.csv")
   
### Step 3: Describing the Data Set ###  

    medical_datafile
    
### Step 4: Converting the String Values into numerical values for Fitting (Vectorisation) ###

    medical_datafile['sex'] = medical_datafile['sex'].apply({'male':0,'female':1}.get) 
    medical_datafile['smoker'] = medical_datafile['smoker'].apply({'yes':1, 'no':0}.get)
    medical_datafile['region'] = medical_datafile['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)
    
### Step 5: Performing Exploratory Data Analysis ###

------Correlations between charges and age-------

    print("Correlation betweeen 'charges' and 'age'")
    sns.jointplot(x=medical_datafile['age'],y=medical_datafile['charges'])
    
------Correlations between charges and smoker-------

    print("Correlation betweeen 'charges' and 'smoker'")
    sns.jointplot(x=medical_datafile['smoker'],y=medical_datafile['charges'])
    
### Step 6: Extracting Features ###

    X = medical_datafile[['age', 'sex', 'bmi', 'children','smoker','region']]
    
### Step 7: Extracting the Predicted Variable ###

    y = medical_datafile['charges']

### Step 8: Displaying the top Features and predicted values ###
    
    X.head()
    y.head()
    
### Step 9: Importing the training and testing model ###

    from sklearn.model_selection import train_test_split

### Step 10: Splitting the training and testing data ###
     
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
     
### Step 11: Importing the linear regression model ###

    from sklearn.linear_model import LinearRegression
    
### Step 12: Fitting the linear model by using the training dataset ###
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    
### Step 13: Predicting the target variable for the linear dataset ###
    
     predictions = model.predict(X_test)
     
### Step 14: Plotting the predictions ###

     * Importing the libraries  
     
     import matplotlib.pyplot as plt
     
     * Plotting a scatter plot
     
    plt.scatter(y_test,predictions)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')

### Step 15: Predicting the charges for a new person ###

     data = {'age' : 30,
        'sex' : 1,
        'bmi' : 35.5,
        'children' : 3,
        'smoker' : 1,
        'region' : 3}
     index = [1]
     Alex = pd.DataFrame(data,index)
     Alex

    prediction_Alex = model.predict(Alex)
    print("Medical Insurance cost for Alex is : ",prediction_Alex)
