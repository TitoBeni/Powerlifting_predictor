import streamlit as st
import pandas as pd
import pickle

st.title("Predicció de moviments")

# Selecció del moviment
movement = st.selectbox('Selecciona el moviment que vols predir:', ['Sentadilla', 'Press de Banca', 'Pes Mort'])

# Carregar el model corresponent segons el moviment seleccionat
if movement == 'Sentadilla':
    with open('Models/random_forest_model_Squat_def.pkl', 'rb') as file:  
        loaded_model = pickle.load(file)
    features = ['Sex', 'Age_6', 'Age_7', 'Age_8', 'Age_9', 'Age_10',
                'BodyweightKg_6', 'BodyweightKg_7', 'BodyweightKg_8', 'BodyweightKg_9', 'BodyweightKg_10',
                'Best3SquatKg_6', 'Best3SquatKg_7', 'Best3SquatKg_8', 'Best3SquatKg_9']
    target = 'Best3SquatKg_10'
elif movement == 'Press de Banca':
    with open('Models/random_forest_model_Bench_def.pkl', 'rb') as file:  
        loaded_model = pickle.load(file)
    features = ['Sex', 'Age_6', 'Age_7', 'Age_8', 'Age_9', 'Age_10',
                'BodyweightKg_6', 'BodyweightKg_7', 'BodyweightKg_8', 'BodyweightKg_9', 'BodyweightKg_10',
                'Best3BenchKg_6', 'Best3BenchKg_7', 'Best3BenchKg_8', 'Best3BenchKg_9']
    target = 'Best3BenchKg_10'
else:  # movement == 'Pes Mort'
    with open('Models/random_forest_model_Deadlift_def.pkl', 'rb') as file:  
        loaded_model = pickle.load(file)
    features = ['Sex', 'Age_6', 'Age_7', 'Age_8', 'Age_9', 'Age_10',
                'BodyweightKg_6', 'BodyweightKg_7', 'BodyweightKg_8', 'BodyweightKg_9', 'BodyweightKg_10',
                'Best3DeadliftKg_6', 'Best3DeadliftKg_7', 'Best3DeadliftKg_8', 'Best3DeadliftKg_9']
    target = 'Best3DeadliftKg_10'

if loaded_model is None:
    st.error("Error carregant el model. Si us plau, comprova el fitxer del model i torna a intentar-ho.")
else:
    def predict_movement(model, data):
        try:
            data_df = pd.DataFrame(data, index=[0])
            prediction = model.predict(data_df[features])
            return prediction
        except Exception as e:
            print(f"Error fent la predicció: {e}")
            return None

    st.write("""
        ## Introdueix les dades per a la predicció:
    """)

    # Tres columnes per a les dades d'entrada de l'usuari
    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.selectbox('Sexe', ['Home', 'Dona'])
        age_6 = st.number_input('Edat fa 4 competicions', 0, 100, 25)
        age_7 = st.number_input('Edat fa 3 competicions', 0, 100, 26)
        age_8 = st.number_input('Edat fa 2 competicions', 0, 100, 27)
        age_9 = st.number_input('Edat anterior', 0, 100, 28)
        age_10 = st.number_input('Edat més recent', 0, 100, 29)

    with col2:
        weight_6 = st.number_input('Pes fa 4 competicions (kg)', 0, 200, 70)
        weight_7 = st.number_input('Pes fa 3 competicions (kg)', 0, 200, 71)
        weight_8 = st.number_input('Pes fa 2 competicions (kg)', 0, 200, 72)
        weight_9 = st.number_input('Pes anterior (kg)', 0, 200, 73)
        weight_10 = st.number_input('Pes més recent (kg)', 0, 200, 74)

    with col3:
        best_3_lift_6 = st.number_input(f'Millors 3 {movement.lower()} fa 4 competicions (kg)', 0, 500, 100)
        best_3_lift_7 = st.number_input(f'Millors 3 {movement.lower()} fa 3 competicions (kg)', 0, 500, 101)
        best_3_lift_8 = st.number_input(f'Millors 3 {movement.lower()} fa 2 competicions (kg)', 0, 500, 102)
        best_3_lift_9 = st.number_input(f'Millors 3 {movement.lower()} anteriors (kg)', 0, 500, 103)
        best_3_lift_10 = st.number_input(f'Millors 3 {movement.lower()} més recents (kg)', 0, 500, 104)

    # Convertir el sexe a valors numèrics (1 per home i 0 per dona)
    sex_numeric = 1 if sex == 'Home' else 0

    # Crear el diccionari de dades
    data = {
        'Sex': sex_numeric,
        'Age_6': age_6,
        'Age_7': age_7,
        'Age_8': age_8,
        'Age_9': age_9,
        'Age_10': age_10,
        'BodyweightKg_6': weight_6,
        'BodyweightKg_7': weight_7,
        'BodyweightKg_8': weight_8,
        'BodyweightKg_9': weight_9,
        'BodyweightKg_10': weight_10,
        'Best3DeadliftKg_6': best_3_lift_6 if movement == 'Pes Mort' else None,
        'Best3DeadliftKg_7': best_3_lift_7 if movement == 'Pes Mort' else None,
        'Best3DeadliftKg_8': best_3_lift_8 if movement == 'Pes Mort' else None,
        'Best3DeadliftKg_9': best_3_lift_9 if movement == 'Pes Mort' else None,
        'Best3DeadliftKg_10': best_3_lift_10 if movement == 'Pes Mort' else None,
        'Best3SquatKg_6': best_3_lift_6 if movement == 'Sentadilla' else None,
        'Best3SquatKg_7': best_3_lift_7 if movement == 'Sentadilla' else None,
        'Best3SquatKg_8': best_3_lift_8 if movement == 'Sentadilla' else None,
        'Best3SquatKg_9': best_3_lift_9 if movement == 'Sentadilla' else None,
        'Best3SquatKg_10': best_3_lift_10 if movement == 'Sentadilla' else None,
        'Best3BenchKg_6': best_3_lift_6 if movement == 'Press de Banca' else None,
        'Best3BenchKg_7': best_3_lift_7 if movement == 'Press de Banca' else None,
        'Best3BenchKg_8': best_3_lift_8 if movement == 'Press de Banca' else None,
        'Best3BenchKg_9': best_3_lift_9 if movement == 'Press de Banca' else None,
        'Best3BenchKg_10': best_3_lift_10 if movement == 'Press de Banca' else None
    }

    # Botó per realitzar la predicció
    if st.button('Fer predicció'):
        prediction = predict_movement(loaded_model, data)
        if prediction is not None:
            st.write(f"La predicció és: {prediction[0]}")
        else:
            st.error("Error fent la predicció. Si us plau, comprova les dades d'entrada i torna a intentar")
