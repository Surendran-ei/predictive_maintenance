import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV

#Import dataset
data = pd.read_csv("CBM Dataset.csv")
df = pd.DataFrame(data)
pd.set_option("display.max_columns",None)
# print(df.info())
# print(df.head())
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.nunique())

#column name transfer to lowercase
df.columns = [col.lower() for col in df.columns]

#Type wise Machine
value = data['Type'].value_counts()
Type_percentage = 100*value/data.Type.shape[0]
labels = Type_percentage.index.array
x = Type_percentage.array
plt.pie(x, labels = labels, autopct='%.0f%%')
plt.title('Machine Type percentage')
plt.show()


#OneHotEncoding for the column type
ohe = OneHotEncoder(sparse_output=False)
encoded_data = ohe.fit_transform(df[["type"]])

encoded_df = pd.DataFrame(encoded_data,columns=ohe.get_feature_names_out(["type"]))

#concatenate the new coulmns
df = pd.concat([df,encoded_df],axis=1)

#drop unwanted columns
# df = df.drop(["udi","product id","type"],axis=1,inplace=False)
df = df.drop(["udi","product id","type","twf","hdf","pwf","osf","rnf"],axis=1,inplace=False)

#correlation checking
corr = df.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr,annot = True)
# plt.show()

#assign the column for X and Y
X = df[['air temperature [k]', 'process temperature [k]', 'rotational speed [rpm]', 'torque [nm]',
        'tool wear [min]','type_H','type_L','type_M']]
Y = df['machine failure']

#Train test split X Y date
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Train the model
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model_1 = RandomForestClassifier()

#fit the model into classifier
model.fit(X_train,Y_train)
model_1.fit(X_train,Y_train)

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
# plt.show()

#prediction
predictions = model.predict(X_test)
prediction_1 = model_1.predict(X_test)
# print("Predictions:", predictions)
# print("actual:",Y_test)

metric_confusion_matrix_output = confusion_matrix(Y_test,predictions)
# print("confusion_matrix",metric_confusion_matrix_output)

classification_output = classification_report(Y_test,predictions)
# print("classification_report", classification_output)

accuracy = accuracy_score(Y_test, predictions)
# print("Decision Tree Classifier accuracy_score\n",accuracy)

accuracy_1 = accuracy_score(Y_test, prediction_1)
# print("Random Forest accuracy_score\n",accuracy_1)





