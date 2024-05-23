# Benchmark models
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV




def grid_lr(X_train, y_train):
    model = LogisticRegression(random_state=666, max_iter=1500)
    solvers = ['liblinear']
    penalty = ['l2','l1',]
    c_values = [ 10, 1.0, 0.01, 0.001,0.0001 ]
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring='f1',error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    return  grid_result.best_estimator_

def grid_SVM(X_train, y_train, performance_metric='f1', resultsGrid=False):
    model = SVC(random_state=666)
    C = np.linspace(0.000001 , 100, 20)
    kernels = ['poly', 'rbf', ]
    gamma = ['scale', 'auto']
    grid = dict(C = C, kernel = kernels, gamma = gamma)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring=performance_metric,error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    if resultsGrid==True:
        return grid_result.cv_results_
    else:
        return  grid_result.best_estimator_
    




def grid_MLP(X_train, y_train):
  model = MLPClassifier(random_state=123)
  hidden_layer_sizes =  [(5, 5)]
  activation = ['logistic']
  solver =  ['sgd'] 
  learning_rate = ['constant', 'invscaling', 'adaptive']
  alpha   =  [ 0.0001, 0.001, 0.01]
  learning_rate_init = [0.0001, 0.001, 0.01, 1]
  batch_size = [X_train.shape[0]]
  momentum = [ 0.8,  0.9 , 1]
  max_iter = [500, 1000, 1500]
  grid = dict(hidden_layer_sizes = hidden_layer_sizes,
              solver = solver,
              alpha = alpha,
              max_iter = max_iter,
              activation = activation,
              batch_size = batch_size,
              learning_rate_init = learning_rate_init,
              momentum = momentum,
              learning_rate = learning_rate)
  cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
  grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                            scoring='f1',error_score='raise')
  grid_result = grid_search.fit(X_train, y_train)
  return  grid_result.best_estimator_


