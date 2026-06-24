import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
# classification part
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#regression part
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

'''
Brown Dwarf -> Star Type = 0
Red Dwarf -> Star Type = 1
White Dwarf-> Star Type = 2
Main Sequence -> Star Type = 3
Supergiant -> Star Type = 4
Hypergiant -> Star Type = 5
'''

# Load and Clean Data
df = pd.read_csv('6 class csv.csv')
df.columns = ['Temperature', 'Luminosity', 'Radius', 'Absolute_Magnitude', 'StarType', 'Color', 'Spectral_Class']
#there is no missing data in the dataset so we can skip that part
df['Color'] = df['Color'].str.lower().str.strip()
df['Color'] = df['Color'].str.split().str.join('-')

X = df[['Temperature', 'Luminosity', 'Radius', 'Color']]
yc = df['StarType']
yr = df['Absolute_Magnitude']

#making transformers
def t1():
    return ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['Color'])],
    remainder='passthrough')     #onehotencoder

def t2():
    return ColumnTransformer([
    ('num', PowerTransformer(method='yeo-johnson'), ['Temperature', 'Luminosity', 'Radius']),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Color'])
])               #power transformer and onehotencoder 

#pipelines for each classification model
pipe1=make_pipeline(
    t1(),
    RandomForestClassifier(random_state=42)
)

pipe2=make_pipeline(
    t1(),
    DecisionTreeClassifier(random_state=42)
)

pipe3=make_pipeline(
    t2(),
    LogisticRegression(max_iter=1000,random_state=42)
)

pipe4=make_pipeline(
    t2(),
    SVC(probability=True,random_state=42)
)

#pipelines for each regression model
pipe5=make_pipeline(
    t1(),
    RandomForestRegressor(random_state=42)
)

pipe6=make_pipeline(
    t1(),
    DecisionTreeRegressor(max_depth=5,min_samples_split=10,min_samples_leaf=4,random_state=42)
)

pipe7=make_pipeline(
    t2(),
    LinearRegression()
)

pipe8=make_pipeline(
    t2(),
    SVR()
)

#cross evaluation for classification models
print(f"Random Forest Mean Accuracy: {cross_val_score(pipe1, X, yc, cv=5).mean() * 100:.2f}%")
print(f"Decision Tree Mean Accuracy: {cross_val_score(pipe2, X, yc, cv=5).mean() * 100:.2f}%")
print(f"Logistic Regression Mean Accuracy: {cross_val_score(pipe3, X, yc, cv=5).mean() * 100:.2f}%")
print(f"SVM Mean Accuracy: {cross_val_score(pipe4, X, yc, cv=5).mean() * 100:.2f}%")

#cross evaluation for regression models
print(f"Random Forest Mean R2: {cross_val_score(pipe5, X, yr, cv=5, scoring='r2').mean():.4f}")
print(f"Decision Tree Mean R2: {cross_val_score(pipe6, X, yr, cv=5, scoring='r2').mean():.4f}")
print(f"Linear Regression Mean R2: {cross_val_score(pipe7, X, yr, cv=5, scoring='r2').mean():.4f}")
print(f"SVM Mean R2: {cross_val_score(pipe8, X, yr, cv=5, scoring='r2').mean():.4f}")

# Fit the models
pipe1.fit(X, yc)
pipe2.fit(X, yc)
pipe3.fit(X, yc)
pipe4.fit(X, yc)

pipe5.fit(X, yr)
pipe6.fit(X, yr)
pipe7.fit(X, yr)
pipe8.fit(X, yr)


model_bundle = {
    'randomforestclassifier': pipe1,
    'decisiontreeclassifier': pipe2,
    'logisticregression': pipe3,
    'svm': pipe4,   
    'randomforestregressor': pipe5,
    'decisiontreeregressor': pipe6,
    'linearregression': pipe7,
    'svr': pipe8
}

# Save the bundle dictionary into one file using Write Binary ('wb') mode
with open('star_models.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)
    
print("Success! 'star_models.pkl' has been created in your workspace.")
