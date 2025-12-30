from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class iqr_clipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns=columns
        self.fences={}
    
    def fit(self, X, y=None):
        for column in self.columns:
            IQ1=X[column].quantile(0.25)
            IQ3=X[column].quantile(0.75)
            IQR=IQ3-IQ1

            lower_fence=IQ1-1.5*IQR
            upper_fence=IQ3+1.5*IQR

            self.fences[column]=(lower_fence, upper_fence)

        return self
        
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            condition=(X[column]<self.fences[column][0]) | (X[column]>self.fences[column][1])
            X[column]=np.where(condition, np.median(X[column]), X[column])
        return X
