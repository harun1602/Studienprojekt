import pandas as pd
import streamlit as st

from KI_Folder.KI_Code import KI_Magic
from data.database_functions import get_user_task_details
from navigation import make_sidebar
import json

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


# Funktion zu korrekten formatieren der Tasks Daten für weiterverarbeitung
def format_task_data(task_data):
    formatted_data = {}
    for version_name, tasks in task_data.items():
        formatted_data[version_name] = []
        for task in tasks:
            task_info = {
                'task_id': task['task_id'],
                'complexity': task['complexity'],
                'total_time': task['total_time'],
                'steps': []
            }
            for step_number, step_time in task['steps']:
                task_info['steps'].append({
                    'step_number': step_number,
                    'step_time': step_time
                })
            formatted_data[version_name].append(task_info)
    return formatted_data


# erstelle Pandas DataFrame
def create_dataframe(calculated_times) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(
        {version: info['calculated_time'] for version, info in calculated_times.items()},
        orient='index',
        columns=['(berechnete) Zeit']
    ).reset_index().rename(columns={'index': 'Version'})

    df['ai_calculated'] = [info['ai_calculated'] for version, info in calculated_times.items()]
    df = df.sort_values(by='Version', ascending=False)
    return df


# Styler für die Tabelle
def highlight_ai_times(row):
    if row['ai_calculated']:
        return ['background-color: green'] * len(row)
    return [''] * len(row)


def admin_ki_prog():

    user_id = st.session_state.current_user.id

    # Aufgabendaten laden
    task_data = get_user_task_details(user_id)

    version_checkboxes = {}

    # Threshold für die Anzahl der benötigten Durchführungen
    threshold = 3

    # Checkboxen für Versionen
    for version_name, tasks in task_data.items():
        st.write(f"**{version_name}: {len(tasks)} Durchführungen**")

        total_tasks = len(tasks)
        # Checkbox nur anzeigen, wenn ausreichend Durchführungen einer Version getätigt wurden
        version_checkboxes[version_name] = st.checkbox(
            f"{version_name} in Prognose einbeziehen",
            value=(total_tasks >= threshold),
            disabled=total_tasks < threshold
        )

    # Prognose starten
    if st.button("Prognose starten", disabled=st.session_state.running, key='run_button'):
        st.session_state.result = None

        selected_versions = [version for version, checked in version_checkboxes.items() if checked]
        if not selected_versions:
            st.warning("Keine Versionen ausgewählt.")
            return

        # Daten der ausgewählten Versionen
        export_data = {}

        formated_data = format_task_data(task_data)

        for version_name in selected_versions:
            export_data[version_name] = formated_data.get(version_name, [])

        # Daten als JSON formatieren
        json_data = json.dumps(export_data, indent=4)
        st.session_state.data = json_data
        st.text("Exportierte Daten:")
        st.text(json_data)


st.title("KI_Prognose")
col_1, col_2 = st.columns([0.6, 0.4]);

with col_1:
    st.subheader("Nutzer und Versionsauswahl")
    admin_ki_prog()

with col_2:
    # Logik zum Anzeigen der Ergebnisse
    result_text = st.empty()
    if st.session_state.result is None:
        result_text.warning("Nach Durchführung der Prognose wird an dieser Stelle das Ergebnis angezeigt")
    else:
        df = create_dataframe(st.session_state.result)
        result_text.write(st.session_state.result)
        st.dataframe(df.style.apply(highlight_ai_times, axis=1))
        st.info("Grün hinterlegte Zeilen sind KI generiert")

    if st.session_state.run_button and st.session_state.data is not None:
        with st.spinner('Daten werden berechnet...'):
            parsed_data = json.loads(st.session_state.data)
            result = KI_Magic(parsed_data)
            st.session_state.result = result
            st.rerun()



