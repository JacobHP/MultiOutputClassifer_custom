import numpy as np
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from joblib import Parallel, delayed
from abc import ABCMeta

def _fit_estimator(estimator, X, y, sample_weight=None):
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


class multi_model(MetaEstimatorMixin,BaseEstimator,metaclass=ABCMeta):
    def __init__(self, hparams, estimators, n_jobs=None):
        """
        hparams: list of parameter dictionaries
        estimators: list of estimators
        """
        self.hparams=hparams #list of hyperparameter dicts
        self.estimators=[estimators[i].set_params(**hparams[i]) for i in range(len(estimators))]
        self.n_jobs=n_jobs

    
    def fit(self, X, y): 
        self.estimators=Parallel(n_jobs=self.n_jobs)(delayed(_fit_estimator)(self.estimators[i],X, y[:,i]) for i in range(len(self.estimators)))
        return self
    
    def predict(self, X):
       y = Parallel(n_jobs=self.n_jobs)(delayed(e.predict)(X) for e in self.estimators)
       return np.asarray(y).T
    
    def predict_proba(self,X):
        results = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.asarray(results)
    
    

