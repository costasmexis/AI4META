# Class balance
from imblearn.over_sampling import SMOTE, BorderlineSMOTE#,ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours

def _class_balance(X, y, bal_type, i=42):
    # Apply class balancing strategies after sfm
    if bal_type == None:
        X_train_selected = X
        y_train = y
    elif bal_type['class_balance'] == 'smote':
        X_train_selected, y_train = SMOTE(random_state=i).fit_resample(X, y)
    elif bal_type['class_balance'] == 'borderline_smote':
        X_train_selected, y_train = BorderlineSMOTE(random_state=i).fit_resample(X, y)
    elif bal_type['class_balance'] == 'tomek':
        tomek = TomekLinks()
        X_train_selected, y_train = tomek.fit_resample(X, y)
    return X_train_selected, y_train