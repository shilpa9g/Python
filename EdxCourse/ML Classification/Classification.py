import csv
import sklearn
from sklearn.svm import SVC
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# reading csv file and extracting class column to y. 
x = pd.read_csv("input3.csv") 
a = np.array(x) 
y  = a[:,2] # classes having 0 and 1 

# extracting two features 
x = np.column_stack((x.A,x.B))

#print(x)
#print(y)

#plt.scatter(x[:, 0], x[:, 1], c=y)
#plt.show()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   x, y, test_size = 0.4, random_state = 1)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

with open('output3.csv', 'w') as g:
    writer = csv.writer(g)

    #SVM Linear
    pipelineSVC = make_pipeline(StandardScaler(), SVC(random_state=1))
    # Create the parameter grid
    param_grid_svc = [{
                        'svc__C': [0.1, 0.5, 1, 5, 10, 50, 100],
                        'svc__kernel': ['linear']
                    }]
    # Create an instance of GridSearch Cross-validation estimator
    gsSVC = GridSearchCV(estimator=pipelineSVC, 
                        param_grid = param_grid_svc,
                        scoring='accuracy',
                        cv=10,
                        refit=True,
                        n_jobs=1)
    # Train the SVM classifier
    gsSVC.fit(x_train, y_train)

    # Print the training score of the best model
    print("\nSVM Linear Training Score", gsSVC.best_score_)

    # Print the model parameters of the best model
    print("SVM Linear Parameters", gsSVC.best_params_)

    # Print the model score on the test data using GridSearchCV score method
    print('SVM Linear Test accuracy: %.4f\n' % gsSVC.score(x_test, y_test))
    output = ['svm_linear', gsSVC.best_score_, gsSVC.score(x_test, y_test)]
    writer.writerow(output)


    #SVM Polynomial
    pipelineSVC = make_pipeline(StandardScaler(), SVC(random_state=1))
    # Create the parameter grid
    param_grid_svc = [{
                        'svc__C': [0.1, 1, 3],
                        'svc__gamma': [0.1, 0.5],
                        'svc__degree': [4, 5, 6],
                        'svc__kernel': ['poly']
                    }]
    # Create an instance of GridSearch Cross-validation estimator
    gsSVC = GridSearchCV(estimator=pipelineSVC, 
                        param_grid = param_grid_svc,
                        scoring='accuracy',
                        cv=5,
                        refit=True,
                        n_jobs=1)
    # Train the SVM classifier
    gsSVC.fit(x_train, y_train)

    # Print the training score of the best model
    print("\nSVM poly Training Score", gsSVC.best_score_)

    # Print the model parameters of the best model
    print("SVM poly Parameters", gsSVC.best_params_)

    # Print the model score on the test data using GridSearchCV score method
    print('SVM poly Test accuracy: %.4f\n' % gsSVC.score(x_test, y_test))
    output = ['svm_polynomial', gsSVC.best_score_, gsSVC.score(x_test, y_test)]
    writer.writerow(output)

    #SVM Polynomial
    from sklearn import metrics
    from sklearn.metrics import classification_report, confusion_matrix
    svcpoly = SVC(kernel='poly', degree=3, C = 0.1, gamma = 0.5)
    svcpoly.fit(x_train, y_train)
    train_test = svcpoly.predict(x_train)
    y_pred = svcpoly.predict(x_test)
    TrAcc = metrics.accuracy_score(y_train, train_test)
    TsAcc = metrics.accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("\nSVM poly Train Accuracy:", TrAcc)
    print("SVM poly Test Accuracy:", TsAcc)
    output = ['svm_polynomial', TrAcc, TsAcc]
    writer.writerow(output)
    

    from sklearn.model_selection import GridSearchCV 
    from sklearn.metrics import classification_report, confusion_matrix
  
    # defining parameter range 
    param_grid = {'C': [0.1, 1, 3],  
                'gamma': [0.1, 0.5], 
                'degree': [4, 5, 6],
                'kernel': ['poly']}  

    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
    
    # fitting the model for grid search 
    grid.fit(x_train, y_train) 
    # print best parameter after tuning 
    print(grid.best_params_) 
    
    # print how our model looks after hyper-parameter tuning 
    print(grid.best_estimator_) 
    grid_predictions = grid.predict(x_test) 
  
    # print classification report 
    print(classification_report(y_test, grid_predictions)) 


    #SVM rbf
    pipelineSVC = make_pipeline(StandardScaler(), SVC(random_state=1))
    # Create the parameter grid
    param_grid_svc = [{
                        'svc__C': [0.1, 0.5, 1, 5, 10, 50, 100],
                        'svc__gamma': [0.1, 0.5, 1, 3, 6, 10],
                        'svc__kernel': ['rbf']
                    }]
    # Create an instance of GridSearch Cross-validation estimator
    gsSVC = GridSearchCV(estimator=pipelineSVC, 
                        param_grid = param_grid_svc,
                        scoring='accuracy',
                        cv=10,
                        refit=True,
                        n_jobs=1)
    # Train the SVM classifier
    gsSVC.fit(x_train, y_train)

    # Print the training score of the best model
    print("\nSVM rbf Training Score", gsSVC.best_score_)

    # Print the model parameters of the best model
    print("SVM rbf Parameters", gsSVC.best_params_)

    # Print the model score on the test data using GridSearchCV score method
    print('SVM rbf Test accuracy: %.4f\n' % gsSVC.score(x_test, y_test))
    output = ['svm_rbf', gsSVC.best_score_, gsSVC.score(x_test, y_test)]
    writer.writerow(output)


    #Logistic Regression
    pipelineLR = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, penalty='l2', solver='lbfgs'))
    # Create the parameter grid
    param_grid_lr = [{
        'logisticregression__C': [0.1, 0.5, 1, 5, 10, 50, 100],
    }]
    #
    # Create an instance of GridSearch Cross-validation estimator
    #
    gsLR = GridSearchCV(estimator=pipelineLR, 
                        param_grid = param_grid_lr,
                        scoring='accuracy',
                        cv=10,
                        refit=True,
                        n_jobs=1)

    # Train the LogisticRegression Classifier
    gsLR = gsLR.fit(x_train, y_train)

    # Print the training score of the best model
    print("LR training accuracy", gsLR.best_score_)

    # Print the model parameters of the best model
    print("LR Parameters", gsLR.best_params_)

    # Print the test score of the best model
    clfLR = gsLR.best_estimator_
    print('LR Test accuracy: %.4f\n' % clfLR.score(x_test, y_test))
    output = ['logistic', gsLR.best_score_, clfLR.score(x_test, y_test)]
    writer.writerow(output)


    #knn classification
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    classifier_knn = KNeighborsClassifier(n_neighbors = 1)
    classifier_knn.fit(x_train, y_train)
    train_test = classifier_knn.predict(x_train)
    y_pred = classifier_knn.predict(x_test)
    TrAcc = metrics.accuracy_score(y_train, train_test)
    TsAcc = metrics.accuracy_score(y_test, y_pred)
    print("knn Train Accuracy:",  TrAcc)
    print("knn Test Accuracy:", TsAcc)
    output = ['knn', TrAcc, TsAcc]
    writer.writerow(output)

    #Decision Trees
    from sklearn import tree
    DTclf = tree.DecisionTreeClassifier()
    DTclf.fit(x_train, y_train)
    train_test = DTclf.predict(x_train)
    y_pred = DTclf.predict(x_test)
    TrAcc = metrics.accuracy_score(y_train, train_test)
    TsAcc = metrics.accuracy_score(y_test, y_pred)
    print("\nDecision Trees Train Accuracy:", TrAcc)
    print("Decision Trees Test Accuracy:", TsAcc)
    output = ['decision_tree', TrAcc, TsAcc]
    writer.writerow(output)

    #Random Forest
    #Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    RFclf = RandomForestClassifier(n_estimators = 10,max_depth = 20,min_samples_split = 6, random_state = 0)
    scores = cross_val_score(RFclf, x_train, y_train, cv = 5)
    train_test = scores.mean()
    print("\nRandom Forest CrossVal Train Accuracy:",  train_test)
    RFclf.fit(x_train, y_train)
    fit_test = RFclf.predict(x_train)
    y_pred = RFclf.predict(x_test)
    TrAcc = metrics.accuracy_score(y_train, fit_test)
    TsAcc = metrics.accuracy_score(y_test, y_pred)
    print("Random Forest Train Accuracy:", TrAcc)
    print("Random Forest Test Accuracy:", TsAcc)
    output = ['random_forest', TrAcc, TsAcc]
    writer.writerow(output)
    