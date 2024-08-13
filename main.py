# Importing libraries 
import numpy as np 
import pandas as pd 
from scipy.stats import mode 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 

# Read the dataset
df = pd.read_csv('dataset/Training.csv')

# Last column is empty, drop
df = df.iloc[:,:-1]

# check diseases
diseases = df['prognosis'].value_counts()

# Encode diseases to integers to be able to make predictions
label_encoder = LabelEncoder()

df['prognosis'] = label_encoder.fit_transform(df['prognosis'])

# Split the data
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Use 10 fold cross validation to check models accuracy
models = {"SVC" : SVC(), "GNB" : GaussianNB(), "RandomForest": RandomForestClassifier()}

for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model,X,y, cv=10, scoring='accuracy')

    print(f"Accuracy of {model_name}: {scores.mean()}")

# create models to manually see accuracy on training and test data
svm_model = SVC()
gaussian_model = GaussianNB()
rf_model = RandomForestClassifier()

svm_model.fit(X_train,y_train)
gaussian_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)

print(f"Accuracy score of svm model on train data: {accuracy_score(y_train,svm_model.predict(X_train))}")
print(f"Accuracy score of gaussian model on train data: {accuracy_score(y_train,gaussian_model.predict(X_train))}")
print(f"Accuracy score of rf model on train data: {accuracy_score(y_train,rf_model.predict(X_train))}")

print(f"Accuracy score of svm model on test data: {accuracy_score(y_test,svm_model.predict(X_test))}")
print(f"Accuracy score of gaussian model on test data: {accuracy_score(y_test,gaussian_model.predict(X_test))}")
print(f"Accuracy score of rf model on test data: {accuracy_score(y_test,rf_model.predict(X_test))}")


# Lastly fit all training data to the last versions of model and predict testing data


final_svm_model = SVC()
final_gaussian_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=0)

final_svm_model.fit(X,y)
final_gaussian_model.fit(X,y)
final_rf_model.fit(X,y)

test_df = pd.read_csv('dataset/Testing.csv')

test_data = test_df.dropna(axis=1)

test_data_X = test_data.iloc[:,:-1]
test_data_y = label_encoder.transform(test_data.iloc[:,-1])

preds_svm_model = final_svm_model.predict(test_data_X)
preds_gaussian_model = final_gaussian_model.predict(test_data_X)
preds_rf_model = final_rf_model.predict(test_data_X)

df_preds = pd.DataFrame(
    {
        'svm': preds_svm_model,
        'gaussian': preds_gaussian_model,
        'rf': preds_rf_model
    }
)

# Take mode of the predictions so that when one model gets a wrong predictions and other two is correct, correct one is chosen
final_predictions = list()
for i, j, k in zip(preds_svm_model, preds_gaussian_model, preds_rf_model):
    final_predictions.append(mode([i,j,k])[0])

print(f"Accuracy on Test dataset by combined model: {accuracy_score(final_predictions,test_data_y)*100}")

confusion_matrix = confusion_matrix(test_data_y,final_predictions)
print("Confusion Matrix:")
print(confusion_matrix)

plt.figure(figsize=(8, 6))
import seaborn as sns  # Make sure seaborn is imported
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

