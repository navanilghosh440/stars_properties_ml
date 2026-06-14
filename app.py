import pandas as pd
import pickle

try:
    with open('star_models.pkl', 'rb') as f:
        bundle = pickle.load(f)

    print('''Available models:(plz use the exact name to select a model.)
          valid input: 'randoM forestSlasSifer', invalid inputs: 'randmfrst classefi'
        To predict star classification:
            -RandomForest Classifier
            -DecisionTree Classifier
            -Logistic Regression
            -SVM
        To predict Absolute Magnitude:
            -RandomForest Regressor
            -DecisionTree Regressor
            -Linear Regression
            -SVR
        Type 'exit' to quit the program.''')

    while True:
        model_choice = ''.join(input("What model would you like to use? ").strip().lower().split())

        if model_choice in bundle:
            model = bundle[model_choice]
            Temperature  = float(input(" -> Enter Temperature (Kelvin): "))
            Luminosity  = float(input(" -> Enter Luminosity (Relative to Sun): "))
            Radius = float(input(" -> Enter Radius (Relative to Sun): "))
            Color  = "-".join(input(" -> Enter Star Color (e.g., Red, Blue): ").lower().strip().split())
            
            df=pd.DataFrame([[Temperature, Luminosity, Radius, Color]], columns=['Temperature', 'Luminosity', 'Radius', 'Color'])
            result = model.predict(df)
            if model_choice in ['randomforestclassifier', 'decisiontreeclassifier', 'logisticregression', 'svm']:
                star_types = {0: 'Brown Dwarf', 1: 'Red Dwarf', 2: 'White Dwarf', 3: 'Main Sequence', 4: 'Supergiant', 5: 'Hypergiant'}
                print(f"Predicted Star Type: {star_types[result[0]]}")
            else:
                print(f"Predicted Absolute Magnitude: {result[0]}")
        elif model_choice.lower() == 'exit':
            print("Exiting the program.")
            break
        else:
            print("Invalid model choice.")
            continue
except FileNotFoundError:
    print("Model file not found. Please run the training script to create the model file before using this program.")
