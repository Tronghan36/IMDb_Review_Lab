import numpy as np
from scipy.sparse import issparse, diags
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

class NBSVMClassifier(BaseEstimator, ClassifierMixin):
    """Mô hình Naive Bayes - Support Vector Machine / Logistic Regression."""
    def __init__(self, alpha=1.0, C=2.0, max_iter=1000, random_state=42):
        self.alpha = alpha
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.r = None
        self.model = None

    def _compute_log_count_ratio(self, X, y):
        X_bin = (X > 0).astype(np.float64)
        p = self.alpha + X_bin[y == 1].sum(axis=0)
        q = self.alpha + X_bin[y == 0].sum(axis=0)
        p_norm = p / np.abs(p).sum()
        q_norm = q / np.abs(q).sum()
        self.r = np.asarray(np.log(p_norm / q_norm)).reshape(-1)

    def fit(self, X, y):
        y = np.asarray(y)
        self._compute_log_count_ratio(X, y)
        r_diag = diags(self.r)
        X_nb = X.dot(r_diag) if issparse(X) else X * self.r
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_nb, y)
        return self

    def predict(self, X):
        r_diag = diags(self.r)
        X_nb = X.dot(r_diag) if issparse(X) else X * self.r
        return self.model.predict(X_nb)

    def predict_proba(self, X):
        r_diag = diags(self.r)
        X_nb = X.dot(r_diag) if issparse(X) else X * self.r
        return self.model.predict_proba(X_nb)

def get_baseline_models(random_state=42):
    """Trả về 6 mô hình Baseline của Phương án 1."""
    return {
        "SVM": LinearSVC(C=1.0, random_state=random_state),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=random_state, n_jobs=-1),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=11, metric="cosine", n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=random_state, n_jobs=-1),
    }

def build_xgb_model(has_gpu=False, random_state=42):
    """Khởi tạo XGBoost tự nhận diện GPU / CPU."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": random_state
    }
    if has_gpu:
        try:
            import xgboost
            if int(xgboost.__version__.split(".")[0]) >= 2:
                params["device"] = "cuda"
            else:
                params["tree_method"] = "gpu_hist"
        except ImportError:
            params["n_jobs"] = -1
    else:
        params["n_jobs"] = -1
    return XGBClassifier(**params)

def compute_soft_voting(probs_dict, weights=[2, 1, 1]):
    """Tính Soft Voting có trọng số cho các xác suất."""
    P = np.column_stack([probs_dict[k] for k in probs_dict])
    w = np.asarray(weights, dtype=float)
    return P @ (w / w.sum())

def build_stacking_model(cv_splits=3, random_state=42):
    """Tạo cấu trúc Stacking Classifier."""
    from sklearn.model_selection import StratifiedKFold
    base_estimators = [
        ("Logistic Regression", LogisticRegression(C=1.0, max_iter=1000, random_state=random_state, n_jobs=-1)),
        ("Naive Bayes", MultinomialNB(alpha=1.0)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=20, random_state=random_state, n_jobs=-1)),
    ]
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=1
    )
