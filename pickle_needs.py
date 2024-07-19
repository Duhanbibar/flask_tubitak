from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)
    

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def fit(self, y):
        self.label_encoder.fit(y)
        return self
    
    def transform(self, y):
        return self.label_encoder.transform(y)
    
    def inverse_transform(self, y):
        return self.label_encoder.inverse_transform(y)

class SVCWithLabelEncoding(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        self.label_encoder = LabelEncoderTransformer()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.classifier.fit(X, y_encoded)
        return self
        
    def predict(self, X):
        y_encoded = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

class XGBCWithLabelEncoding(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        self.label_encoder = LabelEncoderTransformer()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.classifier.fit(X, y_encoded)
        return self
        
    def predict(self, X):
        y_encoded = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)