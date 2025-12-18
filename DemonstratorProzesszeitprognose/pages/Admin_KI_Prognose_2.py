import pandas as pd
import streamlit as st
from KI_Folder.KI_Code_2 import train_new_model_general
from data.database_code import User, Version, session
from data.database_functions import get_user_task_details, load_task_data
from navigation import make_sidebar
from data.filter_dataframe import filter_dataframe
import time as tm
from datetime import date, datetime, time
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import itertools


# --------- Dieser Bereich ist nicht vollständig funktionsfähig und sollte NICHT genutzt werden -----------

st.set_page_config(initial_sidebar_state="expanded", layout="wide")

# Funktion zur Erstellung der Sidebar
make_sidebar()

# Fixiere die Sidebar auf gegebene Größe
st.markdown(
    """
   <style>
   [data-testid="stSidebar"][aria-expanded="true"]{
       min-width: 250px;
       max-width: 250px;
   }
   """,
    unsafe_allow_html=True,
)

# Erstellen nötiger session states
if 'running' not in st.session_state:
    st.session_state.running = False

if 'result' not in st.session_state:
    st.session_state.result = None

if 'data' not in st.session_state:
    st.session_state.data = None

if 'run_button' in st.session_state and st.session_state.run_button:
    st.session_state.running = True
else:
    st.session_state.running = False

if 'df_person_train' not in st.session_state:
    st.session_state.df_person_train = None

if 'df_person_test' not in st.session_state:
    st.session_state.df_person_test = None

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'current_version' not in st.session_state:
    st.session_state.current_version = None

if 'selected_user_name' not in st.session_state:
    st.session_state.all_set = False

if 'selected_version_name' not in st.session_state:
    st.session_state.all_set = False

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if 'user_data_scaling' not in st.session_state:
    st.session_state.user_data_scaling = False

if 'flag_all_data_scaled' not in st.session_state:
    st.session_state.flag_all_data_scaled = False

if 'all_data_scaling' not in st.session_state:
    st.session_state.all_data_scaling = False

if 'flag_all_data_without_target' not in st.session_state:
    st.session_state.flag_all_data_without_target = True

if 'selected_hpo_searcher' not in st.session_state:
    st.session_state.selected_hpo_searcher = None

if 'selected_params' not in st.session_state:
    st.session_state.selected_params = None

if 'sample_weights' not in st.session_state:
    st.session_state.sample_weights = pd.DataFrame.empty


# Funktion zum Zurücksetzen bestimmter Sessions States
def reset_session_state():

    if st.session_state.all_set:
        if st.session_state.current_user != st.session_state.selected_user_name or st.session_state.current_version != st.session_state.selected_version_name:
            st.session_state.current_user = st.session_state.selected_user_name
            st.session_state.current_version = st.session_state.selected_version_name
            st.session_state.df_person_train = None
            st.session_state.df_person_test = None
            st.session_state.data = None
            st.session_state.result = None
    if (st.session_state.selected_user_name is not None and st.session_state.selected_version_name is not None):
        st.session_state.all_set = True


# Umsetzung der Datenvorverarbeitung für Model-Training und Prediction
def format_task_data(task_data_old: pd.DataFrame, version_to_predict):

    # Datensatz kopieren
    task_data = task_data_old.copy()

    # Version, die vorhergesagt werden soll
    x = version_to_predict

    # Nummer für die Version bestimmen
    version_to_predict_num = 1 if x == 'Klemmenleisten-Box Variante 1' else 2 if x == 'Klemmenleisten-Box Variante 2' else 3 if x == 'Klemmenleisten-Box Variante 3' else 4 if x == 'Klemmenleisten-Box Variante 4' else 0

    if version_to_predict_num == 0:
        return -1,-1

    # Anzeige von Formatierungsschritten
    st.write("Rohe Daten")
    st.dataframe(task_data)

    task_data['Start'] = pd.to_datetime(task_data['Start'], format='mixed')
    task_data['Ende'] = pd.to_datetime(task_data['Ende'], format='mixed')

    task_data['Ende_seconds'] = (task_data['Ende'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    task_data['Start_Date'] = task_data['Start'].dt.date
    task_data['Start_Time'] = task_data['Start'].dt.time
    task_data['Ende_Date'] = task_data['Ende'].dt.date
    task_data['Ende_Time'] = task_data['Ende'].dt.time

    task_data['Variante'] = [
        1 if x == 'Klemmenleisten-Box Variante 1' else 2 if x == 'Klemmenleisten-Box Variante 2' else 3 if x == 'Klemmenleisten-Box Variante 3' else 4
        for x in task_data['Version']]

    task_data = task_data.dropna()

    st.write("Start, Ende und Variante hinzugefügt")
    st.dataframe(task_data)

    unique_id_entries = task_data['user_id'].unique()
    for id in unique_id_entries:
        task_data_user = task_data[task_data['user_id'] == id]
        unique_variants_entries = task_data_user['Variante'].unique()
        for variant in unique_variants_entries:
            cnt = 0
            for index, row in task_data.iterrows():
                if row['user_id'] == id and row['Variante'] == variant:
                    task_data.loc[index, "Anzahl montierter gleicher Varianten"] = cnt
                    cnt += 1

    task_data["Anzahl montierter gleicher Varianten"] = task_data["Anzahl montierter gleicher Varianten"].astype(int)

    st.write("Anzahl montierter gleicher Varianten für Nutzer hinzu")
    st.dataframe(task_data)

    #Auf die Person muss hier nicht gefiltert werden, da dies bereits durch Auswahl des Nutzers geschieht.
    df_person = task_data
    df_person = df_person.sort_values(["Ende_seconds"], ascending=[True])

    variant1_cnt = 0
    variant2_cnt = 0
    variant3_cnt = 0
    variant4_cnt = 0

    for index, row in df_person.iterrows():
        variant = row['Variante']
        if variant == 1:
            df_person.loc[index, "Anzahl montierter Variante 1"] = variant1_cnt
            df_person.loc[index, "Anzahl montierter Variante 2"] = variant2_cnt
            df_person.loc[index, "Anzahl montierter Variante 3"] = variant3_cnt
            df_person.loc[index, "Anzahl montierter Variante 4"] = variant4_cnt
            variant1_cnt += 1
        elif variant == 2:
            df_person.loc[index, "Anzahl montierter Variante 1"] = variant1_cnt
            df_person.loc[index, "Anzahl montierter Variante 2"] = variant2_cnt
            df_person.loc[index, "Anzahl montierter Variante 3"] = variant3_cnt
            df_person.loc[index, "Anzahl montierter Variante 4"] = variant4_cnt
            variant2_cnt += 1
        elif variant == 3:
            df_person.loc[index, "Anzahl montierter Variante 1"] = variant1_cnt
            df_person.loc[index, "Anzahl montierter Variante 2"] = variant2_cnt
            df_person.loc[index, "Anzahl montierter Variante 3"] = variant3_cnt
            df_person.loc[index, "Anzahl montierter Variante 4"] = variant4_cnt
            variant3_cnt += 1
        elif variant == 4:
            df_person.loc[index, "Anzahl montierter Variante 1"] = variant1_cnt
            df_person.loc[index, "Anzahl montierter Variante 2"] = variant2_cnt
            df_person.loc[index, "Anzahl montierter Variante 3"] = variant3_cnt
            df_person.loc[index, "Anzahl montierter Variante 4"] = variant4_cnt
            variant4_cnt += 1

    df_person["Anzahl montierter Variante 1"] = df_person["Anzahl montierter Variante 1"].astype(int)
    df_person["Anzahl montierter Variante 2"] = df_person["Anzahl montierter Variante 2"].astype(int)
    df_person["Anzahl montierter Variante 3"] = df_person["Anzahl montierter Variante 3"].astype(int)
    df_person["Anzahl montierter Variante 4"] = df_person["Anzahl montierter Variante 4"].astype(int)

    if not st.session_state.flag_all_data_without_target:
        df_all_data_target = df_person[
                        (df_person['user_id'] != st.session_state.user_id) &
                        (df_person['Variante'] == version_to_predict_num)].copy()

        # Entferne alle Daten, außer "Anzahl montierter Varianten"
        df_all_data_target = df_all_data_target.drop([
            "task_id",
            "user-nachname",
            "user_id",
            "Version",
            "Variante",
            "Revision",
            "Start",
            "Ende",
            "Wahrgenommener Stress",
            "Wahrgenommener Zeitdruck",
            "Wahrgenommene Frustration",
            "Wahrgenommene Komplexität",
            "Spielmodus",
            "Ende_seconds",
            "Start_Date",
            "Start_Time",
            "Ende_Date",
            "Ende_Time",
            # "Anzahl montierter gleicher Varianten",
            # "Anzahl montierter Variante 1",
            # "Anzahl montierter Variante 2",
            # "Anzahl montierter Variante 3",
            # "Anzahl montierter Variante 4"
        ], axis=1)

        df_person = df_person.drop(
                        df_person[
                            (df_person['user_id'] != st.session_state.user_id) &
                            (df_person['Variante'] == version_to_predict_num)
                        ].index
                    )


    # Entferne alle Daten, außer "Anzahl montierter Varianten"
    df_person = df_person.drop([
        "task_id",
        "user-nachname",
        "user_id",
        "Version",
        "Revision",
        "Start",
        "Ende",
        "Wahrgenommener Stress",
        "Wahrgenommener Zeitdruck",
        "Wahrgenommene Frustration",
        "Wahrgenommene Komplexität",
        "Spielmodus",
        "Ende_seconds",
        "Start_Date",
        "Start_Time",
        "Ende_Date",
        "Ende_Time",
        # "Anzahl montierter gleicher Varianten",
        # "Anzahl montierter Variante 1",
        # "Anzahl montierter Variante 2",
        # "Anzahl montierter Variante 3",
        # "Anzahl montierter Variante 4"
    ], axis=1)

    st.write("Anzahl montierter Varianten")
    st.dataframe(df_person)

    df_person_train = df_person.loc[df_person['Variante'] != version_to_predict_num].drop(['Variante'], axis=1)

    # Wenn alle Daten genutzt werden sollen und zusätzlich die Ziel-Daten, dann hier zusammenführen
    if not st.session_state.flag_all_data_without_target:
        df_person_train = pd.concat([df_person_train, df_all_data_target])

    st.dataframe(df_person_train)
    df_person_test = df_person.loc[df_person['Variante'] == version_to_predict_num].drop(['Variante'], axis=1)

    st.dataframe(df_person_test)

    # Rückgabe der formatierten Daten aufgeteilt in train und test
    return df_person_train, df_person_test


# Der Nutzer kann die Einstellungen für die Modelle vornehmen
def model_settings():

    with st.container(border=True):
        st.subheader("Model Einstellungen")

        # Wahl zwischen MLPRegressor und RF
        options_model_algorithm = ['MLP Regressor', 'Random Forest']
        model_algorithm = st.segmented_control("Welcher Grundlegende Algorithmus soll für verwendet werden?", options_model_algorithm, selection_mode="single", on_change=reset_session_state, key='selected_model_algorithm')

        if model_algorithm:

            # Auswahl des Suchraums für die HPO -> Klein, Groß, Custom
            options_params = ['Kleiner Suchraum', 'Großer Suchraum', 'Custom']
            selected_params = st.segmented_control("Welcher Parameterraum soll genutzt werden?", options_params, selection_mode="single",  on_change=reset_session_state, key='selected_params')
            if selected_params:

                # Sollte der Nutzer einen eigenen Suchraum festlegen wollen, dann Einstellungsmöglichkeiten anzeigen für MLPRegressor
                if selected_params == "Custom" and model_algorithm == options_model_algorithm[0]:
                    col1, col2 = st.columns([0.5, 0.5])

                    with col1:
                        # Hidden Layer Parameterbereich
                        lower_bound = st.number_input( "Unteres Ende des Bereichs für Hidden Layer", min_value=1, max_value=100, value=1)

                    with col2:
                        upper_bound = st.number_input("Oberes Ende des Bereichs für Hidden Layer", min_value=1, max_value=100, value=5)

                    if lower_bound > upper_bound:
                        st.error("Das untere Ende muss kleiner oder gleich dem oberen Ende sein.")
                        st.stop()

                    max_layer_size = st.slider("Maximale Anzahl von Layers in einem Tuple", min_value=1, max_value=3, value=2)

                    # Kombinatorik für Hidden Layers generieren
                    st.session_state.param_hidden_layers = []
                    for size in range(1, max_layer_size + 1):
                        st.session_state.param_hidden_layers.extend(itertools.product(range(lower_bound, upper_bound + 1), repeat=size))

                    # Es können die Layer angezeigt werden
                    formatted_layers = ", ".join(str(layer) for layer in st.session_state.param_hidden_layers)
                    with st.expander("Anzeigen der Hidden Layer"):
                        st.write(formatted_layers)

                    # Solver festlegen
                    solver = st.multiselect("Solver auswählen", options=["lbfgs", "sgd", "adam"], default="adam", key="param_solver")

                    # Activation Function auswählen
                    activation = st.multiselect("Activation Function auswählen", options=["identity", "logistic", "tanh", "relu"], default="relu", key="param_activation")

                    # Learning Rate festlegen
                    learning_rate = st.multiselect("Activation Function auswählen", options=[0.1, 0.01, 0.001, 0.0001], default=0.01, key="param_learning_rate")

                    # Scoring Methode festlegen
                    scoring_options = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                         "neg_root_mean_squared_error", "neg_mean_squared_log_error", "neg_root_mean_squared_log_error",
                         "neg_median_absolute_error", "r2", "neg_mean_poisson_deviance", "neg_mean_gamma_deviance",
                         "d2_absolute_error_score"]
                    scoring = st.selectbox("Welcher Scorer soll benutzt werden?", options=scoring_options, index=0, key="param_scoring")

                # Sollte der Nutzer einen eigenen Suchraum festlegen wollen, dann Einstellungsmöglichkeiten anzeigen für RF
                if selected_params == "Custom" and model_algorithm == options_model_algorithm[1]:

                    # Anzahl der Bäume im RandomForest
                    start_n_estimators = st.number_input("Startwert für n_estimators", min_value=10, max_value=5000, value=50, step=100)
                    stop_n_estimators = st.number_input("Stopwert für n_estimators", min_value=10, max_value=5000, value=400, step=100)
                    num_n_estimators = st.number_input("Anzahl der Werte für n_estimators", min_value=1, max_value=50, value=10)

                    st.session_state.n_estimators_forest = [int(x) for x in np.linspace(start=start_n_estimators, stop=stop_n_estimators, num=num_n_estimators)]

                    max_features = st.multiselect("max_features", ["sqrt", "log2", None], default=["log2", None], key="selected_max_features_forest")

                    # Max_depth
                    start_max_depth = st.number_input("Startwert für max_depth", min_value=1, max_value=500, value=10, step=10)
                    stop_max_depth = st.number_input("Stoppwert für max_depth", min_value=1, max_value=500, value=80, step=10)
                    num_max_depth = st.number_input("Anzahl der Werte für max_depth", min_value=1, max_value=50,value=11)
                    max_depth = [int(x) for x in
                                 np.linspace(start=start_max_depth, stop=stop_max_depth, num=num_max_depth)]
                    max_depth.append(None)

                    st.session_state.max_depth_forest = max_depth

                    # Min_Depth
                    min_samples_split = st.multiselect("min_samples_split", [2, 5, 10], default=[2, 5, 10], key="selected_min_samples_split")

                    min_samples_leaf = st.multiselect("min_samples_leaf", [1, 2, 4], default=[1, 2, 4], key="selected_min_samples_leaf")

                    bootstrap = st.multiselect("bootstrap", [True, False], default=[True, False], key="bootstrap_forest")

                    # Scoring Methode festlegen
                    scoring_options = ["explained_variance", "max_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                                       "neg_root_mean_squared_error", "neg_mean_squared_log_error", "neg_root_mean_squared_log_error",
                                       "neg_median_absolute_error", "r2", "neg_mean_poisson_deviance",
                                       "neg_mean_gamma_deviance","d2_absolute_error_score"]

                    scoring = st.selectbox("Welcher Scorer soll benutzt werden?", options=scoring_options, index=0, key="param_scoring_forest")

                # Auswahl des HPO Searchers
                options_hpo_searcher = ['GridSearchCV', 'RandomizedSearchCV']
                hpo_searcher = st.segmented_control("Welcher Searcher soll für die HPO verwendet werden?", options_hpo_searcher, selection_mode="single", on_change=reset_session_state, key='selected_hpo_searcher')

                if hpo_searcher:

                    # Falls RandomizedSearch ausgewählt weitere Einstellungsmöglichkeiten anzeigen
                    if hpo_searcher == options_hpo_searcher[1]:
                        
                        n_iter = st.number_input(
                            "Anzahl der durchzuführenden Kombinationen",
                            min_value=1, max_value=1000, value=50, key='param_n_iter', step=20
                        )

                        cv_option = st.selectbox("Cross Validation Methode (Achtung: Deutliche Verlängerung der Berechnungzeit (x20))", options=["LeaveOneOut", "KFold", None], index=2,  key="selected_cv_option")

                    # Es können weitere Einstellungen zu den übergebenen Daten festgelegt werden
                    st.write("Einstellungen zu den genutzten Daten:")

                    # Vorerst nicht umgesetzt
                    #user_data_scaled = st.toggle("Nutzer-Daten skalieren", key='flag_user_data_scaled')
                    #if user_data_scaled:
                    #    col1, col2 = st.columns([0.04, 0.96])
                    #    with col2:
                    #        user_data_scaling = st.selectbox("Wie sollen die User-Daten skaliert werden?", ['2:1','3:1', '3:2', '4:1'], key='user_data_scaling', index=0, on_change=reset_session_state)


                    all_data = st.toggle("Daten anderer Nutzer einbeziehen", key='flag_all_data')
                    if all_data:
                        col1, col2 = st.columns([0.04, 0.96])
                        with col2:

                            # Vorerst nicht umgesetzt
                            #all_data_scaled = st.toggle("Daten anderer Nutzer skalieren", key='flag_all_data_scaled')
                            #if all_data_scaled:
                            #    all_data_scaling = st.selectbox("Wie sollen die User-Daten skaliert werden?", ['1:2','1:3', '2:3', '1:4'], key='all_data_scaling', index=0, on_change=reset_session_state)

                            all_data_without_target = st.toggle("Daten anderer Nutzer, ohne Ziel Version", value= True, key='flag_all_data_without_target')


# Prognose Einstellungen vornehmen
def admin_ki_prog():

    with st.container(border=True):

        st.subheader("Nutzer und Versionsauswahl")

        # Benutzer auswählen
        user_options = session.query(User).all()
        user_dict = {f"{user.firstname} {user.lastname}": user.id for user in user_options}
        user_name = st.selectbox("KI-Prognose durchführen für:", list(user_dict.keys()), key='selected_user_name', index=1, on_change=reset_session_state)
        user_id = user_dict[user_name]

        st.session_state.user_id = user_id

        versions = session.query(Version).all()

        # Liste über alle Versionsnamen
        version_names = [version.name for version in versions]
        version_names.append("Unbekannte Version")
        # Selectbox zum Auswählen der gewünschten Version
        selected_version_name = st.selectbox("KI-Prognose durchführen für Version:", version_names, on_change=reset_session_state, key='selected_version_name')

        # Aufgabendaten laden
        task_data = load_task_data()

        if user_id and selected_version_name:

            st.write("Resultierender Aufgabenpool:")
            # DataFrame auf den ausgewählten Nutzer filtern
            filtered_data_with_version = task_data[task_data['user_id'] == user_id]

            filtered_data = filtered_data_with_version.copy()

            event = st.dataframe(
                filtered_data,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
            )

        with st.container(border=True):

            col_1, col_2 = st.columns([0.5, 0.5]);

            # Anzeigen der ausgewählten Daten
            with col_1:
                st.write("Zum Export ausgewählte Aufgaben:")

                tasks_to_export = event.selection.rows
                filtered_df = filtered_data.iloc[tasks_to_export]
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )

            with col_2:
                # Gruppieren der ausgewählten Tasks nach Version und Anzahl berechnen
                if not filtered_df.empty:
                    version_counts = filtered_df['Version'].value_counts().reset_index()
                    version_counts.columns = ['Version', 'Anzahl der Tasks']
                    st.write("Anzahl der ausgewählten Tasks pro Version:")
                    st.dataframe(version_counts, use_container_width=True, hide_index=True)
                else:
                    st.write("Keine ausgewählten Tasks")


    model_settings()

    # Prognose starten
    if st.button("Prognose starten", disabled=st.session_state.running or (st.session_state.selected_model_algorithm is None or st.session_state.selected_hpo_searcher is None or st.session_state.selected_params is None or st.session_state.flag_all_data is None), key='run_button'):
        st.session_state.result = None

        if filtered_df.empty:
            st.warning("Wähle Aufgaben zur Prognose!")
            return

        else:


            if st.session_state.flag_all_data:

                all_user_task_data = task_data[task_data['user_id'] != user_id]
                
                if st.session_state.flag_all_data_without_target:
                    all_user_task_data = all_user_task_data[all_user_task_data['Version'] != selected_version_name]

                # Vorerst nicht umgesetzt
                #if st.session_state.all_data_scaling:
                #    filtered_df = pd.concat([filtered_df] * 2)  # Doppelte Gewichtung

                concat_df = pd.concat([filtered_df,all_user_task_data])

                # Daten formatieren und in Train und Test aufteilen
                df_person_train, df_person_test = format_task_data(concat_df, selected_version_name)

                # Vorerst nicht umgesetzt
                #if st.session_state.flag_user_data_scaled:

                    # Standard-Sample-Weights (einfach gewichtet)
                    #sample_weights = np.ones(len(df_person_train))

                    # Setze den ersten Eintrag auf doppelte Gewichtung
                    #sample_weights[0] = 2

                    #sample_weights[len(filtered_df) // 2] = 3

                    # Setze den letzten Eintrag auf dreifache Gewichtung
                    #sample_weights[len(filtered_df) - 1] = 3

                    #st.session_state.sample_weights = sample_weights

                    #print(f"Länge Sample: {len(sample_weights)}")

            else:
                df_person_train, df_person_test = format_task_data(filtered_df, selected_version_name)

            st.session_state.df_person_train = df_person_train
            st.session_state.df_person_test = df_person_test


# Anzeigen der Prognose Ergebnisse
def ki_result():

    if st.session_state.data is not None:
        st.dataframe()

    # Logik zum Anzeigen der Ergebnisse
    result_text = st.empty()
    if st.session_state.result is None:
        result_text.warning("Nach Durchführung der Prognose wird an dieser Stelle das Ergebnis angezeigt")
    else:

        result_text.text(st.session_state.result)
        if st.session_state.result is not None:

            # Erstelle ein Liniendiagramm mit results
            if st.session_state.user_id is not None:
                task_data = load_task_data()
                filtered_data_with_version = task_data[task_data['user_id'] == st.session_state.user_id]
                filtered_data = filtered_data_with_version[filtered_data_with_version['Version'] == st.session_state.selected_version_name]
                filtered_data.reset_index(drop=True, inplace=True)
                fig = px.line(st.session_state.result, title='Vorhersage vs. Tatsächlich', labels={'x': 'Index', 'y': 'Wert'})
                fig.data[0].name = "Vorhergesagte Werte"
                fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Zeit'], mode='lines', name='Tatsächliche Zeiten'))
            else:
                fig = px.line(st.session_state.result)
            
            st.plotly_chart(fig, use_container_width=True)


    if st.session_state.run_button and st.session_state.df_person_train is not None and st.session_state.df_person_test is not None:

        # Falls keine zulässige Version ausgewählt wurde
        if not isinstance(st.session_state.df_person_train, pd.DataFrame):
            if st.session_state.df_person_train == -1:
                st.warning("Für die Variante können wir aktuell keine Prognose fällen")
        # Spinner anzeigen, solange Berechnung durchgeführt werden
        else:
            with st.spinner('Daten werden berechnet...'):

                #if not st.session_state.flag_user_data_scaled:
                #    st.session_state.sample_weights = None

                st.session_state.result = train_new_model_general(st.session_state.df_person_train,
                                                                  st.session_state.df_person_test,
                                                                  st.session_state.selected_model_algorithm,
                                                                  st.session_state.selected_hpo_searcher,
                                                                  #st.session_state.flag_user_data_scaled,
                                                                  False,
                                                                  #st.session_state.user_data_scaling,
                                                                  False,
                                                                  st.session_state.flag_all_data,
                                                                  #st.session_state.flag_all_data_scaled,
                                                                  False,
                                                                  #st.session_state.all_data_scaling,
                                                                  False,
                                                                  st.session_state.selected_version_name,
                                                                  st.session_state.flag_all_data_without_target,
                                                                  st.session_state.selected_user_name,
                                                                  st.session_state.selected_params,
                                                                  #st.session_state.sample_weights
                                                                  None)

                st.rerun()


def show_pred_and_real():
    if isinstance(st.session_state.df_person_train, pd.DataFrame):
        st.dataframe(st.session_state.df_person_train)
    #if isinstance(st.session_state.df_person_train, pd.DataFrame):
    #    st.dataframe(st.session_state.df_person_test)
    st.dataframe(st.session_state.result)

st.title("KI_Prognose")
col_1, col_2 = st.columns([0.6, 0.4]);

with col_1:
    admin_ki_prog()

with col_2:
    with st.container(border=True):
        st.subheader("KI - Prognose")
        with st.container(border=True):
            ki_result()

        if st.session_state.df_person_test is not None:
            with st.container(border=True):
                show_pred_and_real()
