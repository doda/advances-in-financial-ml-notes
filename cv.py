import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import log_loss, accuracy_score

def getTrainTimes(t1, testTimes):
    trn = t1.copy(deep=True)
    for i, j in testTimes.iteritems():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  # Train starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index  # Train ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index  # Train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes(times, pctEmbargo):
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1], index=times[-step:]))
    return mbrg

def cvScore(clf, X, y, sample_weight=None, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=None):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('Wrong scoring method')
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    if sample_weight is None:
        sample_weight = np.ones(len(X))

    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :], y=y.iloc[train],
            sample_weight=sample_weight[train]
        )
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight[test], labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight[test])
        score.append(score_)
    return np.array(score)


class PurgedKFold(_BaseKFold):
    '''
    Extend KFold to work with labels that span intervals
    The train is is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training examples in between
    '''
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label through Dates must be a pandas series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo
        
    def split(self, X, y=None, groups=None):
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError('X and ThruDateValues must have the same index length')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices
