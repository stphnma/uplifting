import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
import numpy as np


class ClassTransform():
    '''
    implements the class-transform model as defined https://arxiv.org/abs/1504.01132
    '''
    
    def __init__(self, propensity_model=None, uplift_model=None):
        # TODO: check that the models follow sklearn API
        if propensity_model is None:
            self.propensity_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
        else:
            self.propensity_model = propensity_model
            
        if uplift_model is None:
            self.uplift_model = Ridge(alpha=1e3, copy_X=False, fit_intercept=False)
        else:
            self.uplift_model = uplift_model
    
    def fit(self, X, y, W):
        '''
        X: array-like, feature vector
        y: array-like, vector of outcomes
        W: array-like, vector of treatment indicators
        '''
        # Fit propensity score model
        self.propensity_model.fit(X, W)    
        p = self.propensity_model.predict_proba(X)[:, 1]
        y_star = y * (W - p) / (p * (1-p))
        self.uplift_model.fit(X, y_star)
        
    def predict(self, X):
        return self.uplift_model.predict(X)


def plot_deciles(y, y_star, treated, bins=20):

    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y_star, pd.Series):
        y_star = y_star.values
    if isinstance(treated, pd.Series):
        treated = treated.values
    
    sorted_idx = np.argsort(-1* y_star)  # sort in descending
    y_star = y_star[sorted_idx]
    y = y[sorted_idx]
    treated = treated[sorted_idx]

    deciles = []
    size = len(y)//bins

    for i, pos in enumerate(range(0, len(y), size)):
        y_ = y[pos:min(pos + size, len(y))]
        y_star_ = y_star[pos:min(pos + size, len(y))]
        treated_ = treated[pos:min(pos + size, len(y))]

        pred_avg = np.mean(y_star_)

        actual_avg = np.sum(y_[np.where(treated_==1)]) / float(len(y_[np.where(treated_==1)])) - \
                        np.sum(y_[np.where(treated_==0)]) / float(len(y_[np.where(treated_==0)]))

        deciles.append({
            'actual': actual_avg,
            'predicted': pred_avg,
            'decile': i
        })
        
    deciles = pd.DataFrame(deciles).set_index('decile')
    deciles.plot(kind='bar'); plt.show()
        

if __name__ == '__main__':
    # df = pd.read_pickle('revenue/concessions.p').reset_index(drop=True)
    # df = pd.read_csv('/Users/sma/Downloads/criteo-uplift.csv')
    df = pd.read_csv('criteo-uplift-sampled.csv')
    X = df.drop(['treatment','conversion', 'exposure', 'visit'], axis=1)
    W = df['treatment']
    y = df['visit']

    upl = ClassTransform()
    upl.fit(X, y, W)
    y_star = upl.predict(X)
    plot_deciles(y, y_star, W, bins=10)


    