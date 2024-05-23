import joblib
import pandas as pd
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, RocCurveDisplay


df = pd.read_excel('Reduced_Features_for_TAI_project.xlsx')

# separating the 0 and 1 labels just in case it's useful later
df_negatives = df[df['Label'] == 0] # healthy
df_positives = df[df['Label'] == 1] # lesion
labels_GT = df[['Patient ID', 'Label']].copy() # saving the original labels of the micros to compare to later

df.drop(columns=['Patient ID', 'Label'], axis=1, inplace=True) # dropping the Patient ID and Label columns as we don't want to give that information to the model


# df is X, GT_labels is Y
testSize = 0.2
X_train, X_test, y_train, y_test = train_test_split(df, labels_GT, test_size=testSize, random_state=100) # splitting on train and test (trying with 0.1)
y_train = y_train.iloc[:, -1] # leaving out the id of the labels
y_test = y_test.iloc[:, -1]


# ----------------------------------------------------------- K-Nearest Neighbor -------------------------------------------------------

# Initializing the SVM classifier
svm_classifier = svm.SVC()

# Trying to select again the significant features to see if it returns the same ones or not
selector = k_best_selector = SelectKBest(score_func=f_classif, k=75)
selector.fit(X_train, y_train)

selected_features = X_train.columns[selector.get_support()]

X_train_selected = selector.transform(X_train) # reduce X to the selected features
X_test_selected = selector.transform(X_test)


# Defining the parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],  # Relevant for 'poly' kernel
    'coef0': [0.0, 0.1, 0.5]  # Relevant for 'poly' and 'sigmoid' kernels
}

# Performing Grid Search
best_random = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)



# ------------------------------------------------------ Training ----------------------------------------------

# Fit the Grid Search to the data
best_random.fit(X_train_selected, y_train)

# Get the best parameters and best estimator
best_params = best_random.best_params_
best_estimator = best_random.best_estimator_

# Predict the labels of the test set using the best model
y_pred_BESVM = best_random.predict(X_test_selected)

# Saving the model so it doesn't need to be fitted again -> saving the best parameters and best estimator found by hte randomized search
joblib.dump(best_random, "SVM_best_estimator.keras")
joblib.dump(best_params, "SVM_best_params.keras")

# Base model for Decision Trees
SVM_classifier_base = svm.SVC()
SVM_classifier_base.fit(X_train_selected, y_train)
y_pred = SVM_classifier_base.predict(X_test_selected)


# Getting accuracy of both models to compare
accuracy_base = accuracy_score(y_test, y_pred)
print(f"Accuracy base: {accuracy_base}")

accuracy_best = accuracy_score(y_test, y_pred_BESVM)
print(f"Accuracy best: {accuracy_best}")

print(f'Improvement of {round((100 * (accuracy_best - accuracy_base) / accuracy_base), 3)}% with the randomized search for parameters')


conf_mat = confusion_matrix(y_test, y_pred_BESVM)
print(f"Confusion Matrix:\n{conf_mat}\n\n")

# classification report
print(classification_report(y_test, y_pred_BESVM, labels=[0, 1]))#, target_names=['Healthy', 'Lesion']))

# getting the best parameters in a string to be saved in the ROC curve figure
parameters_name = []
for key, value in best_params.items():
    if key == 'gamma' or key == 'kernel': key=''
    parameters_name.append(f"{key}{str(value)}")

parameters_name = '_'.join(parameters_name)
print(parameters_name)

# ROC curve
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(best_random, X_test_selected, y_test, ax=ax, alpha=0.8)
plt.title(f"ROC SVM\n{parameters_name}")
plt.savefig(f"Results/RF_{parameters_name}")