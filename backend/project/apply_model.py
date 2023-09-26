from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
def ApplyModel(x_train, y_train):
    random_forest = RandomForestClassifier(n_estimators=200, max_depth=100)
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42) # set extra tree model
    knn_model = KNeighborsClassifier(n_neighbors=3) # set KNN model
    des_tree = DecisionTreeClassifier(random_state=42) # set decision tree model
    random_forest.fit(x_train, y_train)
    et_model.fit(x_train, y_train)
    knn_model.fit(x_train, y_train)
    des_tree.fit(x_train, y_train)
    ensemble_model = VotingClassifier(estimators=[ # ensemble model with voting classifier
    ('Random Forest', random_forest), # Random Forest model
    ('Extra Trees', et_model), # Extra Trees model
    ("Decision Tree", des_tree), # Decision Tree model
    ("KNN", knn_model) # KNN model
], voting='hard',weights=[1,1,1,1])
    ensemble_model.fit(x_train, y_train)
    return ensemble_model
    
    
def Train_model(x_train_sm, y_train_sm):
    random_forest = ApplyModel(x_train_sm,  y_train_sm)
    return random_forest