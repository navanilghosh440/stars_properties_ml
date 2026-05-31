import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# classification part
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#regression part
from sklearn.metrics import mean_absolute_error, r2_score
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

# Split the data into training and testing sets
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.2, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, yr, test_size=0.2, random_state=42)

#making transformers
t1=ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Color'])
    ], remainder='passthrough')

t2=ColumnTransformer([
    ('num', StandardScaler(), ['Temperature', 'Luminosity', 'Radius']),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Color'])
])

#pipelines for each classification model
pipe1=make_pipeline(
    t1,
    RandomForestClassifier(random_state=42)
)

pipe2=make_pipeline(
    t1,
    DecisionTreeClassifier(random_state=42)
)

pipe3=make_pipeline(
    t2,
    LogisticRegression(max_iter=1000,random_state=42)
)

pipe4=make_pipeline(
    t2,
    SVC(random_state=42)
)

#pipelines for each regression model
pipe5=make_pipeline(
    t1,
    RandomForestRegressor(random_state=42)
)
pipe6=make_pipeline(
    t1,
    DecisionTreeRegressor(max_depth=5,min_samples_split=10,min_samples_leaf=4,random_state=42)
)

pipe7=make_pipeline(
    t2,
    LinearRegression()
)
pipe8=make_pipeline(
    t2,
    SVR()
)

# Fit the models
pipe1.fit(Xc_train, yc_train)
pipe2.fit(Xc_train, yc_train)
pipe3.fit(Xc_train, yc_train)
pipe4.fit(Xc_train, yc_train)

pipe5.fit(Xr_train, yr_train)
pipe6.fit(Xr_train, yr_train)
pipe7.fit(Xr_train, yr_train)
pipe8.fit(Xr_train, yr_train)

if __name__ == "__main__":
    #make predictions and evaluate accuracy for classification models
    predictions1 = pipe1.predict(Xc_test)
    predictions2 = pipe2.predict(Xc_test)
    predictions3 = pipe3.predict(Xc_test)
    predictions4 = pipe4.predict(Xc_test)

    print(f"Random Forest Classification Accuracy: {accuracy_score(yc_test, predictions1) * 100:.2f}%")
    print(f"Decision Tree Classification Accuracy: {accuracy_score(yc_test, predictions2) * 100:.2f}%")
    print(f"Logistic Regression Accuracy: {accuracy_score(yc_test, predictions3) * 100:.2f}%")
    print(f"SVM Accuracy: {accuracy_score(yc_test, predictions4) * 100:.2f}%")

    #make predictions and evaluate accuracy for regression models
    predictions5 = pipe5.predict(Xr_test)
    predictions6 = pipe6.predict(Xr_test)
    predictions7 = pipe7.predict(Xr_test)
    predictions8 = pipe8.predict(Xr_test)

    print(f"Random Forest MAE: {mean_absolute_error(yr_test, predictions5)}, R2 Score: {r2_score(yr_test, predictions5)}")
    print(f"Decision Tree MAE: {mean_absolute_error(yr_test, predictions6)}, R2 Score: {r2_score(yr_test, predictions6)}")
    print(f"Linear Regression MAE: {mean_absolute_error(yr_test, predictions7)}, R2 Score: {r2_score(yr_test, predictions7)}")
    print(f"SVM MAE: {mean_absolute_error(yr_test, predictions8)}, R2 Score: {r2_score(yr_test, predictions8)}")

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