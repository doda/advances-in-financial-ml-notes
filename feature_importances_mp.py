from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score

from mlfinlab.feature_importance import (
    feature_importance_mean_decrease_impurity,
    feature_importance_mean_decrease_accuracy,
    feature_importance_sfi,
)

from mlfinlab.cross_validation import ml_cross_val_score, PurgedKFold
from mlfinlab.util.multiprocess import process_jobs
from sklearn.model_selection import KFold


def feature_importances(X, cont, method, allow_masking_effects=False, n_splits=10):
    max_features = None if allow_masking_effects else 1
    clf = DecisionTreeClassifier(
        criterion='entropy', max_features=max_features, class_weight='balanced', min_weight_fraction_leaf=0.0
    )
    clf = BaggingClassifier(
        base_estimator=clf, n_estimators=1000, max_features=1.0, max_samples=1.0, oob_score=True, n_jobs=-1
    )
    fit = clf.fit(X, cont['bin'])
    oob_score = fit.oob_score_

    cv_gen = PurgedKFold(n_splits=n_splits, samples_info_sets=cont['t1'])
    oos_score = ml_cross_val_score(clf, X, cont['bin'], cv_gen=cv_gen, scoring=accuracy_score).mean()

    if method == 'MDI':
        imp = feature_importance_mean_decrease_impurity(fit, X.columns)
    elif method == 'MDA':
        imp = feature_importance_mean_decrease_accuracy(clf, X, cont['bin'], cv_gen, scoring=accuracy_score)
    elif method == 'SFI':
        imp = feature_importance_sfi(clf, X, cont['bin'], cv_gen, scoring=accuracy_score)
    
    return imp, oob_score, oos_score
