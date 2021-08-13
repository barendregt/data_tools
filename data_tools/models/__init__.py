"""
The ``data_tools.models`` module provides custom made models
"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, clone
from math import ceil

class BalancedStackEnsembleClassifier(BaseEstimator):
    def __init__(self, base_estimator = None, verbosity = None):
        self._classes = []
        self._estimators = {}
        self._class_ratios = {}
        self._base_estimator = base_estimator
        self._verbosity = verbosity
        self._min_sample_threshold = 100
        
    def fit(self, X, y):        
        # Find distinct classes in target and initialize estimator dict
        self._classes = np.unique(y)
        self._estimators = {cl: [] for cl in self._classes}
        
        # Determine imbalance per class and build class-specific ensembles
        n_total = y.shape[0]
        for _class in self._classes:
            positive_size = (y==_class).sum()
            negative_size = n_total-(y==_class).sum()
            
            if positive_size >= negative_size: # Positive class is larger
                X_major_class = pd.DataFrame(X[y==_class])
                X_minor_class = pd.DataFrame(X[y!=_class])               
                
                y_major_class = np.array([1]*X_major_class.shape[0])
                y_minor_class = np.array([0]*X_minor_class.shape[0])
            else: # Negative class is larger
                X_minor_class = pd.DataFrame(X[y==_class])
                X_major_class = pd.DataFrame(X[y!=_class])
                
                y_major_class = np.array([0]*X_major_class.shape[0])
                y_minor_class = np.array([1]*X_minor_class.shape[0])       
                
            for ii in range(0, X_major_class.shape[0], X_minor_class.shape[0]):
                start_ii = ii
                stop_ii  = ii+X_minor_class.shape[0]
                
                if (stop_ii >= X_major_class.shape[0]) & ((X_major_class.shape[0] - ii) > self._min_sample_threshold):
                    stop_ii = X_major_class.shape[0]                    
                elif ((X_major_class.shape[0] - ii) <= self._min_sample_threshold):
                    continue
                
                _X = pd.concat([X_major_class[start_ii:stop_ii], X_minor_class.sample(frac=1)[0:(stop_ii-start_ii)]], axis=0, ignore_index=True)
                _y = np.hstack([y_major_class[start_ii:stop_ii], y_minor_class[0:(stop_ii-start_ii)]])
                                
                _mdl = clone(self._base_estimator)
                  
                self._estimators[_class].append(_mdl.fit(_X, _y))

    def predict_proba(self, X):
        
        probabilities = pd.DataFrame([])
      
        # Loop over classes and predict using class-specific models
        for _class in self._classes:
            _class_probabilities = np.array([0.0]*X.shape[0])
            for _model in self._estimators[_class]:
                _class_probabilities += _model.predict_proba(X)[:,1]
            # Average probality per class
            probabilities[_class] = _class_probabilities / len(self._estimators[_class])     
            
        predicted_probs = probabilities.values / probabilities.values.max(axis=0)
        
        return predicted_probs
                
    def predict(self, X):
     
        predicted_probs = self.predict_proba(X)
        
        # Final estimator        
        predicted_label = np.array(self._classes)[np.argmax(predicted_probs, axis=1)]                
  
        return predicted_label  