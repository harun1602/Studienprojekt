import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import streamlit as st


import logging
import time

# --------- Dieser Bereich ist nicht vollständig funktionsfähig und sollte NICHT genutzt werden -----------

# Konfiguration für das Logging in app.log
logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)


# Methode zum Laden des durch den Nutzer ausgewählten Modells
def load_model(model_algorithm):

    if model_algorithm == "MLP Regressor":

        # Rückgabe eines "Basis" Models. Die Parameter werden ggf später überschrieben
        return MLPRegressor(
            activation='relu',
            solver='adam',
            batch_size=1,
            max_iter=10000,
            shuffle=True,
            random_state=1,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            hidden_layer_sizes=[(3, 2), (3, 3), (3, 4), (3, 5),]
        )
    if model_algorithm == 'Random Forest':
        return RandomForestRegressor()
    else:
        return None

# Methode zum Laden des HPO_Searcher
def load_searcher(model, hpo_searcher, selected_params):

    # Unterscheidung zwischen MLPRegressor und RandomForest
    # Falls der Nutzer den MLP Regressor ausgewählt hat
    if isinstance(model, MLPRegressor):

        scoring = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                   "neg_root_mean_squared_error", "neg_mean_squared_log_error", "neg_root_mean_squared_log_error",
                   "neg_median_absolute_error", "r2", "neg_mean_poisson_deviance", "neg_mean_gamma_deviance",
                   "d2_absolute_error_score"]

        scorer = scoring[0]

        # Festlegung einer CrossValidation Methode
        cv_options = ["LeaveOneOut", "KFold"]
        cv = None
        if st.session_state.selected_cv_option == cv_options[0]:
            cv = LeaveOneOut()
        if st.session_state.selected_cv_option == cv_options[1]:
            cv = KFold(n_splits=5, shuffle=True, random_state=9)


        # Falls der Nutzer einen kleinen Suchraum ausgewählt hat
        if selected_params == "Kleiner Suchraum":

            parameters = {"hidden_layer_sizes": [
                (1,), (2,), (3,), (4,), (5,),
                (2, 2), (2, 3),
                (3, 2), (3, 3), (3, 4), (3, 5),
                (4, 2), (4, 3), (4, 4), (4, 5),
                (5, 2), (5, 3), (5, 4), (5, 5),
                (3, 2), (4, 2), (5, 2),
                (2, 3), (4, 3), (5, 3),
                (2, 4), (3, 4), (5, 4),
                (2, 5), (3, 5), (4, 5),
            ],
                "learning_rate_init": [0.1, 0.01, 0.001]
            }

        # Falls der Nutzer einen Große Suchraum ausgewählt hat
        if selected_params == "Großer Suchraum":

            parameters = {"hidden_layer_sizes": [
                (1,), (2,), (3,), (4,), (5,),
                (2, 2), (2, 3),
                (3, 2), (3, 3), (3, 4), (3, 5),
                (4, 2), (4, 3), (4, 4), (4, 5),
                (5, 2), (5, 3), (5, 4), (5, 5),
                (3, 2), (4, 2), (5, 2),
                (2, 3), (4, 3), (5, 3),
                (2, 4), (3, 4), (5, 4),
                (2, 5), (3, 5), (4, 5),
            ],
                "learning_rate_init": [0.1, 0.01, 0.001],
                "solver": ['adam', 'lbfgs', 'sgd'],
                "activation": ['identity', 'logistic', 'tanh', 'relu']
            }

        # Falls der Nutzer einen eigenen Parameter Raum festgelegt hat
        if selected_params == "Custom":
            parameters = {
                "hidden_layer_sizes": st.session_state.param_hidden_layers,
                "learning_rate_init": st.session_state.param_learning_rate,
                "solver": st.session_state.param_solver,
                "activation": st.session_state.param_activation
            }
            scorer = st.session_state.param_scoring

        # Rückgabe des jeweiligen HPO Searchers mit den durch den Nutzer festgelegten Parametern
        if hpo_searcher == "GridSearchCV":
            return GridSearchCV(model, parameters, n_jobs=-1, verbose=3, scoring=scorer, cv=cv)

        if hpo_searcher == "RandomizedSearchCV":
            scoring = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_squared_log_error", "neg_root_mean_squared_log_error", "neg_median_absolute_error", "r2", "neg_mean_poisson_deviance", "neg_mean_gamma_deviance", "d2_absolute_error_score"]
            return RandomizedSearchCV(model, parameters, n_iter=st.session_state.param_n_iter, n_jobs=-1, random_state=8, verbose=3, scoring=scorer, cv= cv)


    # Falls der RandomForestRegressor gewählt wurde
    if isinstance(model, RandomForestRegressor):

        # Scoring Methoden
        scoring = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                   "neg_root_mean_squared_error", "neg_mean_squared_log_error", "neg_root_mean_squared_log_error",
                   "neg_median_absolute_error", "r2", "neg_mean_poisson_deviance", "neg_mean_gamma_deviance",
                   "d2_absolute_error_score"]

        scorer = scoring[0]

        # Festlegen der CrossValidation Methode
        cv_options = ["LeaveOneOut", "KFold"]
        cv = None
        if st.session_state.selected_cv_option == cv_options[0]:
            cv = LeaveOneOut()
        if st.session_state.selected_cv_option == cv_options[1]:
            cv = KFold(n_splits=5, shuffle=True, random_state=9)

        # Kleiner Suchraum
        if selected_params == "Kleiner Suchraum":

            # Anzahl Bäume im random forest
            n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
            # Anzahl der Feature, die zum Split berücksichtigt werden
            max_features = ['log2', 'sqrt']
            # Maximum Tiefe eines Baums
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum Anzahl der samples, die zum Split berücksichtigt werden
            min_samples_split = [2, 5, 10]
            # Minimale Anzahl der samples für Blatt
            min_samples_leaf = [1, 2, 4]

            bootstrap = [True]
            # Parameter festlegen
            parameters = {'n_estimators': n_estimators,
                          'max_features': max_features,
                          'max_depth': max_depth,
                          'min_samples_split': min_samples_split,
                          'min_samples_leaf': min_samples_leaf,
                          'bootstrap': bootstrap}

        # Falls der große Suchraum für den RF gewählt wurde
        if selected_params == "Großer Suchraum":

            #Anzahl Bäume im random forest
            n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
            # Anzahl der Feature, die zum Split berücksichtigt werden
            max_features = ['log2', 'sqrt']
            # Maximum Tiefe eines Baums
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum Anzahl der samples, die zum Split berücksichtigt werden
            min_samples_split = [2, 5, 10]
            # Minimale Anzahl der samples für Blatt
            min_samples_leaf = [1, 2, 4]

            bootstrap = [True, False]
            # Paramter festlegen
            parameters = {'n_estimators': n_estimators,
                          'max_features': max_features,
                          'max_depth': max_depth,
                          'min_samples_split': min_samples_split,
                          'min_samples_leaf': min_samples_leaf,
                          'bootstrap': bootstrap}

        if selected_params == "Custom":

            # Erstelle den Parameter-Pool mit Nutzerauswahl
            parameters = {
                "n_estimators": st.session_state.n_estimators_forest,
                "max_features": st.session_state.selected_max_features_forest,
                "max_depth": st.session_state.max_depth_forest,
                "min_samples_split": st.session_state.selected_min_samples_split,
                "min_samples_leaf": st.session_state.selected_min_samples_leaf,
                "bootstrap": st.session_state.bootstrap_forest
            }
            
            scorer = st.session_state.param_scoring_forest

        # Rückgabe des jeweiligen HPO Searchers mit den durch den Nutzer festgelegten Parametern
        if hpo_searcher == "GridSearchCV":
            return GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1, verbose=3, scoring=scorer, cv=cv)


        if hpo_searcher == "RandomizedSearchCV":
            return RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=st.session_state.param_n_iter, random_state=8, n_jobs=-1, verbose=3, scoring=scorer, cv=cv)


    return None

# Methode zum Trainieren der neuen (durch den Nutzer) festgelegten Modelle.
def train_new_model_general(df_person_train: pd.DataFrame, df_person_test: pd.DataFrame, model_algorithm, hpo_searcher, user_data_scaled, user_data_scaling, all_data, all_data_scaled, all_data_scaling, target_version, flag_all_data_without_target, selected_user, selected_params, sample_weights):

    # Logging der übergebenen Parameter
    logging.info(f"Training gestartet: Modell={model_algorithm}, Suchalgorithmus={hpo_searcher}, Nutzer={selected_user}, Alle Daten Flag={all_data}, ohne Ziel-Variante={flag_all_data_without_target}, Ziel-Variante={target_version}, user_data_scaling={user_data_scaling}, sample_weights={sample_weights is not None}, selected_params={selected_params}")

    # Startzeit messen
    start_time = time.time()

    X_train = df_person_train.drop(["Zeit"], axis=1)
    y_train = df_person_train["Zeit"]

    X_test = df_person_test.drop(["Zeit"], axis=1)
    y_test = df_person_test["Zeit"]

    # Daten Skalieren
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # gewähltes Model laden
    model = load_model(model_algorithm)
    if model is None:
        logging.error(f"Unbekannter Algorithmus: {model_algorithm}")
        return
    # gewählten Searcher laden
    searcher = load_searcher(model, hpo_searcher, selected_params)
    if searcher is None:
        logging.error(f"Unbekannter Searcher: {hpo_searcher}")
        return

    # Neues Model trainieren
    best = searcher.fit(X_train, y_train)

    # Endzeit messen
    end_time = time.time()

    # Dauer berechnen
    elapsed_time = end_time - start_time
    
    # Logging der besten Parameter + Zeiten
    print(f'Finished! Model: {model_algorithm}; HPO-Searcher: {hpo_searcher}')
    logging.info(f"Training abgeschlossen: Modell={model_algorithm}, Suchalgorithmus={hpo_searcher}")
    logging.info(f"Benötigte Zeit: {elapsed_time:.2f} Sekunden")
    logging.info(f"Beste Parameter: {best.best_params_}")
    
    # Skalieren der vorherzusagenden Daten
    X_test = scaler.transform(X_test)
    
    # Vorhersage mit besten Parametern durchführen
    y_pred = best.predict(X_test)
    
    # Berechnen von MSE und R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Logging
    logging.info(f"MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Rückgabe der Vorhersagen
    return y_pred

