import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from data.database_code import engine, session, User, Task, Version
from data.database_functions import load_task_data, load_task_steps, get_users
from data.filter_dataframe import filter_dataframe
from data.statistics_functions import average_task_completion_time_per_version_panda, \
    std_dev_task_completion_time_per_version_panda, average_perceived_complexity_per_version_for_user, \
    average_perceived_complexity_per_version_for_all
from navigation import make_sidebar
import plotly.graph_objects as go

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


def task_details():

    # Lade alle Aufgaben aus der Datenbank
    task_data = load_task_data()

    # Wenn Aufgaben vorhanden
    if not task_data.empty:
        st.subheader("Aufgabenübersicht")
        filtered_task_data = filter_dataframe(task_data)

        # Lege Anzahl der zu sehenden Rows fest
        numRows = min(7, len(filtered_task_data.index))
        st.dataframe(filtered_task_data, hide_index=True, height=(numRows + 1) * 35 + 3)

        # Task-Auswahl
        selected_task_id = st.selectbox("Wähle eine Aufgabe", options=filtered_task_data['task_id'])

        if selected_task_id:
            st.subheader("Montageschritte der ausgewählten Aufgabe")
            task_steps_data = load_task_steps(selected_task_id)
            if not task_steps_data.empty:
                st.dataframe(task_steps_data, hide_index=True)
            else:
                st.write("Keine Montageschritte für diese Aufgabe gefunden.")
    # Wenn keine Aufgaben gefunden: Ausgabe
    else:
        st.warning("Es konnten keine Aufgaben gefunden werden.")


def display_average_times_version(tasks_df: pd.DataFrame, versions_df: pd.DataFrame):

    # Anzeigen der durchschnittlichen Zeit pro Version

    st.subheader("Durchschnittliche Zeit pro Version")

    avg_time_by_version = average_task_completion_time_per_version_panda(tasks_df, versions_df)

    st.write("")
    st.write("")

    st.dataframe(avg_time_by_version)

    fig = px.bar(avg_time_by_version, x='Version', y='Durchschnittliche Zeit',
                 title='Durchschnittliche Aufgabenzeit pro Version',
                 labels={'Durchschnittliche Zeit': 'Durchschnittliche Aufgabenzeit (Sekunden)',
                         'Version': 'Version'})
    st.plotly_chart(fig, use_container_width=True)

# Darstellen der Standardabweichung
def display_standard_deviation(tasks_df: pd.DataFrame, versions_df: pd.DataFrame, tasks_with_versions: pd.DataFrame):
    st.subheader("Standardabweichung pro Version")

    st.dataframe(std_dev_task_completion_time_per_version_panda(tasks_df, versions_df))

    # Berechne den Mittelwert und die Standardabweichung der Aufgabenzeit pro Version
    std_mean_grouped = tasks_with_versions.groupby('name')['time'].agg(['mean', 'std']).reset_index()

    # Erstelle Error-Balken Diagramm bei der die Standardabweichung mit angezeigt wird
    fig_errorbar = px.bar(std_mean_grouped,
                          x='mean',
                          y='name',
                          orientation='h',
                          error_x='std',
                          labels={'mean': 'Durchschnittliche Aufgabenzeit (Sekunden)', 'name': 'Version'},
                          title='Durchschnittliche Aufgabenzeit mit Standardabweichung pro Version')

    st.plotly_chart(fig_errorbar, use_container_width=True)

# Darstellen der wahrgenommenen Komplexität nach Nutzer
def display_perceived_complexity_user(tasks_df: pd.DataFrame, versions_df: pd.DataFrame):
    st.subheader("Gefühlte Komplexität nach User")
    users = session.query(User).all()

    if users:
        # Nutzerauswahl
        user_list = [f"{user.firstname} {user.lastname} ({user.nickname})" for user in users]

        selected_user = st.selectbox("Welcher Nutzer soll betrachtet werden?", user_list, index=None, placeholder="Wähle einen Nutzer!", key='selected_user')

        if selected_user:
            selected_user_obj = users[user_list.index(st.session_state.selected_user)]
            # Berechnen der gefühlten Komplexität füt den Nutzer
            st.dataframe(average_perceived_complexity_per_version_for_user(tasks_df, versions_df, selected_user_obj.id), hide_index=True)

    else:
        st.warning("Es konnten keine Nutzer gefunden werden")


def display_box_plot_versions():
    # Daten aus der Datenbank abfragen
    tasks = session.query(Task.id, Task.time, Task.version_id, Version.name).join(Version).all()

    # DataFrame erstellen
    tasks_df = pd.DataFrame(tasks, columns=['task_id', 'time', 'version_id', 'version_name'])

    # Sortiere DataFrame
    tasks_df = tasks_df.sort_values(by='version_name')

    # Boxplot erstellen
    fig_completion_times = px.box(tasks_df, x='version_name', y='time',
                                  title='Verteilung der Bearbeitungszeiten nach Version')

    st.subheader("Box-Plots")

    # Diagramm anzeigen
    st.plotly_chart(fig_completion_times, use_container_width=True)


# Darstellen der wahrgenommenen Komplexität für alle Nutzer
def display_perceived_complexity_all(tasks_df: pd.DataFrame, versions_df: pd.DataFrame):

    st.subheader("Wahrgenommene Komplexität")

    avg_complex_by_version = average_perceived_complexity_per_version_for_all(tasks_df, versions_df)

    st.dataframe(avg_complex_by_version, hide_index=True)


# Darstellen der Lernkurve eines Nutzers
def display_learning_curve_user(tasks_with_versions: pd.DataFrame, users_df: pd.DataFrame):
    def scale_complexity_to_user_rating(complexity):

        # Allgemein von x in Bereich [a,b] auf [c,d]: ((x-a)/(b-a))*(d-c) + c

        # Transformation von [1, 10] auf [1, 4]
        return ((complexity - 1) / (10-1)) * (4-1) + 1

    st.subheader("Lernkurve des Nutzers")

    users = session.query(User).all()

    # Nutzerauswahl
    user_list = [f"{row.firstname} {row.lastname} ({row.nickname})" for row in users_df.itertuples()]

    selected_user_learning_curve = st.selectbox("Wähle einen Nutzer", user_list, index=None, placeholder="Wähle einen Nutzer!", key='selected_user_learning_curve')
    # Versionsauswahl
    selected_version_learning_curve = st.selectbox("Wähle eine Version", versions_df['name'].unique(), index=None, placeholder="Wähle eine Version")

    # Wenn Nutzer und Version ausgewählt wurde
    if selected_user_learning_curve and selected_version_learning_curve:
        selected_user_obj = users[user_list.index(st.session_state.selected_user_learning_curve)]
        selected_user_id = selected_user_obj.id

        user_data = tasks_with_versions[(tasks_with_versions['user_id'] == selected_user_id) &
                                        (tasks_with_versions['name'] == selected_version_learning_curve)]

        if not user_data.empty:
            # Erstelle ein Liniendiagramm mit ausgewählten Daten
            fig = px.line(user_data, x="id", y=['time', 'perceived_complexity'],
                          title=f'Lernkurve für Nutzer "{selected_user_learning_curve}" und Version "{selected_version_learning_curve}"',
                          labels={'value': 'Wert', 'index': 'Aufgaben ID'},
                          line_shape='linear',
                          markers=True,
                          hover_data={
                              'start_timestamp': True  # Timestamp beim Hover anzeigen
                          }
                          )

            # Skaliere die tatsächliche Komplexität
            user_data['scaled_complexity'] = user_data['complexity'].apply(lambda x: scale_complexity_to_user_rating(x))

            # Füge die tatsächliche Komplexität als konstante Linie hinzu
            fig.add_scatter(x=user_data['id'], y=user_data['scaled_complexity'], mode='lines', name='Skalierte Tatsächliche Komplexität')

            # Anzeigen der IDs als category, sodass keine Gleitkommazahlen entstehen
            fig.update_xaxes(type='category')
            fig.update(layout_yaxis_range=[0, None])

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No Data found!")


def display_histo():
    st.subheader('Häufigkeitsverteilung nach Version')

    df = load_task_data()

    if not df.empty:

        df = df.sort_values(by='Version')
        # Mittelwert und Standardabweichung berechnen
        mean = df['Zeit'].mean()
        stdev = df['Zeit'].std()
        stdev_plus = mean + stdev
        stdev_minus = mean - stdev

        # Min- und Max-Werte für die x-Achse definieren
        #min_time = df['Zeit'].min()
        #max_time = df['Zeit'].max()

        #max_count = df['Zeit'].max()

        # Plotly Histogramm erstellen
        fig = px.histogram(
            df,
            x='Zeit',
            color='Version',
            title='Häufigkeitsverteilung der Prozesszeiten nach Versionen',
            labels={'Zeit': 'Prozesszeit', 'Version': 'Version'},
            nbins=30,  # Anzahl der Bins
            marginal='violin' # One of 'rug', 'box', 'violin', or 'histogram',

        )

        # Standardabweichung + Durchschnitt als Linie anzeigen
        #fig.add_shape(type="line", x0=mean, x1=mean, y0=0, y1=1, xref='x', yref='y', line=dict(color='blue', dash='dash'), name='Durchschnitt', showlegend=True)

        # Zeige Linie, nur wenn Standardabweichung berechnet
       # if not np.isnan(stdev):
       #     fig.add_shape(type="line", x0=stdev_plus, x1=stdev_plus, y0=0, y1=1, xref='x', yref='y', line=dict(color='red', dash='dash'), name='Standardabweichung', showlegend=True)
        #    fig.add_shape(type="line", x0=stdev_minus, x1=stdev_minus, y0=0, y1=1, xref='x', yref='y', line=dict(color='red', dash='dash'), visible=True)

        # Layout des Histo anpassen
        fig.update_layout(
            xaxis_title='Prozesszeit',
            yaxis_title='Häufigkeit',
            # barmode='overlay',
            barmode='group',
            bargap=0.05,
            # bargroupgap=0.1
            #xaxis=dict(range=[0, None])

        )
        fig.update_xaxes(
            rangemode='nonnegative'
        )

        st.plotly_chart(fig)

    else:
        st.warning("Es konnten keine Daten gefunden werden")


# Line Chart zur Verdeutlichung der Auswirkungen der Spielmodi
def line_chart_game_mode_stats(curr_data):

    if not curr_data.empty:
        curr_data_copy = curr_data.copy()
        curr_data_copy.drop('version', axis=1, inplace=True)

        # Durchschnittliche Werte pro Spielmodus berechnen
        average_metrics = curr_data_copy.groupby('game_mode').mean().reset_index()

        # Umstrukturieren der Daten
        average_metrics_melted = average_metrics.melt(id_vars='game_mode',
                                                      value_vars=['perceived_complexity', 'perceived_stress',
                                                                  'perceived_time_pressure', 'perceived_frustration'],
                                                      var_name='Metric',
                                                      value_name='Average')

        # Umbenennung der Metriken für bessere Lesbarkeit
        average_metrics_melted['Metric'] = average_metrics_melted['Metric'].replace({
            'perceived_complexity': 'Perceived Complexity (PC)',
            'perceived_stress': 'Perceived Stress (PS)',
            'perceived_time_pressure': 'Perceived Time Pressure (PTP)',
            'perceived_frustration': 'Perceived Frustration (PF)'
        })

        # Liniendiagramm erstellen
        fig = px.line(average_metrics_melted,
                      x='Metric',
                      y='Average',
                      color='game_mode',
                      title='Durchschnittliches User Feedback nach Game Mode',
                      labels={'Metric': 'User Impression Categories', 'Average': 'Average Value', 'game_mode': 'Game Mode'})

        fig.update_layout(
            yaxis=dict(range=[0, 4])
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Es konnten keine Daten gefunden werden")

# RadarChart zur Darstellung des Nutzer-Feedbacks
def radar_chart_game_mode_stats(curr_data):

    if not curr_data.empty:
        curr_data_copy = curr_data.copy()
        curr_data_copy.drop('version', axis=1, inplace=True)

        # Durchschnittliche Werte pro Spielmodus berechnen
        average_metrics = curr_data_copy.groupby('game_mode').mean().reset_index()

        # Umstrukturieren der Daten
        average_metrics_melted = average_metrics.melt(id_vars='game_mode',
                                                      value_vars=['perceived_complexity', 'perceived_stress',
                                                                  'perceived_time_pressure', 'perceived_frustration'],
                                                      var_name='Metric',
                                                      value_name='Average')

        # Umbenennung der Metriken für bessere Lesbarkeit
        average_metrics_melted['Metric'] = average_metrics_melted['Metric'].replace({
            'perceived_complexity': 'Perceived Complexity (PC)',
            'perceived_stress': 'Perceived Stress (PS)',
            'perceived_time_pressure': 'Perceived Time Pressure (PTP)',
            'perceived_frustration': 'Perceived Frustration (PF)'
        })

        # Daten für Radar Chart vorbereiten
        categories = ['Perceived Complexity (PC)', 'Perceived Stress (PS)', 'Perceived Time Pressure (PTP)',
                      'Perceived Frustration (PF)']
        fig = go.Figure()

        for mode in average_metrics['game_mode']:
            fig.add_trace(go.Scatterpolar(
                r=average_metrics[average_metrics['game_mode'] == mode].iloc[0, 1:],  # Durchschnittswerte pro Metrik
                theta=categories,
                fill='toself',
                name=mode
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 4]
                )),
            dragmode=False,
            modebar_remove='zoom',
            title="Radar Chart User Feedback nach Game Mode"
        )

        # Rescale der Achsen verhindern
        #fig.layout.xaxis.fixedrange = True
        #fig.layout.yaxis.fixedrange = True

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Es konnten keine Daten gefunden werden")


# Funktion zur Darstellung von Box Plots für das Feedback der User über die Game-Modes
def display_perceived_by_gamemode(curr_data, metrics):

    curr_data_copy = curr_data.copy()

    metric, label = metrics

    selected_version = st.selectbox("Wähle eine Version", versions_df['name'].unique(), placeholder="Wähle eine Version", key=f'display_{metric}', index=1 if versions_df.size > 1 else 0)

    if selected_version and not curr_data_copy.empty:

        filtered_df = curr_data_copy[curr_data_copy['version'] == selected_version]

        fig = px.box(filtered_df, x='game_mode', y=metric,
                     title=f'{label} Distribution nach Game Mode',
                     labels={'game_mode': 'Game Mode', metric: label})

        fig.update(layout_yaxis_range=[0, 4])
        st.plotly_chart(fig, use_container_width=True)


def corr_display(curr_data):

    # Korrelationsmatrix berechnen
    correlation_matrix = curr_data.corr()

    # Heatmap der Korrelationsmatrix erstellen
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", zmin=-1, zmax=1, color_continuous_scale='rdylgn', title='Heatmap zur Correlation')
    st.plotly_chart(fig, use_container_width=True)


def display_feedback_history():

    st.subheader("Nutzer Feedback Verlauf")

    users = session.query(User).all()

    # Nutzerauswahl
    user_list = [f"{row.firstname} {row.lastname} ({row.nickname})" for row in users_df.itertuples()]

    selected_user_feedback_history = st.selectbox("Wähle einen Nutzer", user_list, index=None,
                                                placeholder="Wähle einen Nutzer!", key='selected_user_feedback_history')
    # Versionsauswahl
    selected_version_feedback_history = st.selectbox("Wähle eine Version", versions_df['name'].unique(), index=None,
                                                   placeholder="Wähle eine Version", key='selected_version_feedback_history')

    # Wenn Nutzer und Version ausgewählt wurde
    if selected_user_feedback_history and selected_version_feedback_history:
        selected_user_obj = users[user_list.index(st.session_state.selected_user_feedback_history)]
        selected_user_id = selected_user_obj.id

        user_data = tasks_with_versions[(tasks_with_versions['user_id'] == selected_user_id) &
                                        (tasks_with_versions['name'] == selected_version_feedback_history)]

        if not user_data.empty:
            # Erstelle ein Liniendiagramm mit ausgewählten Daten
            fig = px.line(user_data, x="id", y=['perceived_complexity','perceived_stress', 'perceived_time_pressure', 'perceived_frustration'],
                          title=f'Feedback Verlauf für Nutzer "{selected_user_feedback_history}" und Version "{selected_version_feedback_history}"',
                          labels={'value': 'Wert', 'index': 'Aufgaben ID'},
                          line_shape='linear',
                          markers=True)

            # Füge die tatsächliche Komplexität als konstante Linie hinzu
            #fig.add_scatter(x=user_data['id'], y=user_data['complexity'], mode='lines+markers', name='Tatsächliche Komplexität')

            # Anzeigen der IDs als category, sodass keine Gleitkommazahlen entstehen
            fig.update_xaxes(type='category')

            st.plotly_chart(fig, use_container_width=True)



def display_learning_curve_multi_user(tasks_with_versions: pd.DataFrame, users_df: pd.DataFrame):

    st.subheader("Lernkurve mehrerer Nutzer")

    # Nutzerauswahl
    user_list = [f"{row.firstname} {row.lastname} ({row.nickname})" for row in users_df.itertuples()]

    users_dict = {f"{row.firstname} {row.lastname} ({row.nickname})": row.id for row in users_df.itertuples()}  # Mapping Name zu ID


    selected_users_learning_curve = st.multiselect("Wähle einen oder mehrere Nutzer", user_list, default=None, placeholder="Wähle Nutzer!", key='selected_multi_user_learning_curve')
    # Versionsauswahl
    selected_version_learning_curve = st.selectbox("Wähle eine Version", versions_df['name'].unique(), index=None, placeholder="Wähle eine Version", key='selected_version_multi_user_learning_curve')

    if selected_users_learning_curve and selected_version_learning_curve:

        # Leerer DataFrame für alle Nutzerdaten
        combined_data = pd.DataFrame()

        for user in selected_users_learning_curve:

            # ID über Dictionary holen
            selected_user_id = users_dict[user]

            # Daten des Nutzers und der Version filtern
            user_data = tasks_with_versions[(tasks_with_versions['user_id'] == selected_user_id) &
                                            (tasks_with_versions['name'] == selected_version_learning_curve)]

            if not user_data.empty:

                # Neue Spalte fü Anzahl der Durchläufe
                user_data = user_data.copy()
                user_data['run'] = range(1, len(user_data) + 1)

                # Hinzufügen des Nutzernamens, um später nach Nutzer zu filtern
                user_data['user'] = user

                combined_data = pd.concat([combined_data, user_data], ignore_index=True)

        if not combined_data.empty:

            # Liniendiagramm für alle Nutzer erstellen
            fig = px.line(combined_data,
                          x="run",
                          y=['time'],
                          color='user',
                          title=f'Lernkurve für ausgewählte Nutzer und Version "{selected_version_learning_curve}"',
                          labels={'run': 'Durchlauf', 'value': 'Wert'},
                          line_shape='linear',
                          markers=True,
                          hover_data={
                              'start_timestamp': True  # Timestamp beim Hover anzeigen
                          }
                          )

            fig.update_xaxes(type='category')
            fig.update_layout(yaxis_range=[0, None])

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Data found!")



# Lade Daten in DataFrames
tasks_df = pd.read_sql('SELECT * FROM tasks', engine)
versions_df = pd.read_sql('SELECT id, name, complexity FROM versions', engine)
users_df = pd.read_sql('SELECT id, firstname, lastname, nickname FROM users', engine)


# Verbinde tasks mit versions, um die Versionsnamen hinzuzufügen
tasks_with_versions = pd.merge(tasks_df, versions_df, left_on='version_id', right_on='id', suffixes=('', '_version'))

st.title("Statistik-Seite")

# -------- Design der Seite (Aufbau und Aussehen)  --------
height_row_one = 700
height_row_two = 680

c1_1, c1_2 = st.columns([0.5, 0.5])
c2_1, c2_2 = st.columns(2)
c3_1, c3_2 = st.columns(2)
c4_1, c4_2 = st.columns(2)
c5_1, c5_2 = st.columns(2)
c6_1, c6_2 = st.columns(2)
c7_1, c7_2 = st.columns(2)
c8_1, c8_2 = st.columns(2)
c9_1, c9_2 = st.columns(2)

if not tasks_df.empty:
    with st.container():
        with c1_1:
            with st.container(height=height_row_one, border=True):
                task_details()
        with c1_2:
            with st.container(height=height_row_one, border=True):
                display_average_times_version(tasks_df, versions_df)


    with st.container():
        with c2_1:
            with st.container(border=True, height= height_row_two):
                display_standard_deviation(tasks_df, versions_df, tasks_with_versions)
        with c2_2:
            with st.container(border=True, height= height_row_two):
                display_box_plot_versions()

    with st.container():
        with c3_1:
            with st.container(border=True):
                display_perceived_complexity_user(tasks_df, versions_df)

        with c3_2:
            with st.container(border=True):
                display_perceived_complexity_all(tasks_df, versions_df)

    with st.container():
        with c4_1:
            with st.container(border=True):
                display_learning_curve_user(tasks_with_versions, users_df)
        with c4_2:
            with st.container(border=True):
                display_histo()


    with st.container():

        tasks_data = session.query(Task).all()
        data = pd.DataFrame([{
            'user_id': task.user_id,
            'game_mode': task.game_mode,
            'perceived_complexity': task.perceived_complexity,
            'perceived_stress': task.perceived_stress,
            'perceived_time_pressure': task.perceived_time_pressure,
            'perceived_frustration': task.perceived_frustration,
            'real_complexity': task.version.complexity,
            'version': task.version.name,
        } for task in tasks_data])

        metrics = {
            'perceived_complexity': 'Perceived Complexity',
            'perceived_stress': 'Perceived Stress',
            'perceived_time_pressure': 'Perceived Time Pressure',
            'perceived_frustration': 'Perceived Frustration'
        }

        with c5_1:
            with st.container(border=True):
                display_perceived_by_gamemode(data, metrics.popitem())

        with c5_2:
            with st.container(border=True):
                display_perceived_by_gamemode(data, metrics.popitem())

        with c6_1:
            with st.container(border=True):
                display_perceived_by_gamemode(data, metrics.popitem())

        with c6_2:
            with st.container(border=True):
                display_perceived_by_gamemode(data, metrics.popitem())

        with c7_1:
            with st.container(border=True):
                line_chart_game_mode_stats(data)

        with c7_2:
            with st.container(border=True):
                radar_chart_game_mode_stats(data)

        with c8_1:

            with st.container(border=True):

                if not data.empty:

                    st.subheader("Korrelation des User -Feedbacks")

                    selected_gamemode = st.selectbox("Spielmodus", options=["classic", "timer", "countdown"])

                    if selected_gamemode:

                        filtered_data = data[data['game_mode'] == selected_gamemode]

                        corr_data = filtered_data[['perceived_complexity', 'perceived_stress', 'perceived_time_pressure', 'perceived_frustration', 'real_complexity']]

                        if not corr_data.empty:
                            corr_display(corr_data)
                        else:
                            st.warning("No Data")

        with c8_2:
            with st.container(border=True):
                display_feedback_history()

        with c9_1:
            with st.container(border=True):
                display_learning_curve_multi_user(tasks_with_versions, users_df)

else:
    st.warning("Es konnten keine Aufgaben-Daten gefunden werden. Bitte führen Sie zunächst einige Aufgaben im Arbeitsplatz durch!")

