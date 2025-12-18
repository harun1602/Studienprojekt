import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from data.database_code import engine, session, User, Task, Version
from data.database_functions import load_task_data, load_task_steps
from data.filter_dataframe import filter_dataframe
from data.statistics_functions import average_task_completion_time_per_version_panda, \
    std_dev_task_completion_time_per_version_panda, average_perceived_complexity_per_version_for_user
from navigation import make_sidebar


def arbeitsverlauf_page():

    # Setzen des Seitenlayouts
    st.set_page_config(
        page_title="ProScheduler: Verlauf",
        initial_sidebar_state="expanded",
        layout="centered")

    # Einladen der Sidebar für die Naviagtion zwischen den Seiten
    make_sidebar()

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

    # Wenn eingeloggt, dann öffne die Seite (Else fall kann eigentlich nicht eintreten, da mit Logout erneut auf die Anmelde-Seite geleitet)
    if 'logged_in' in st.session_state and st.session_state.logged_in:

        # Ziehe den aktuell angemeldeten User
        current_user = st.session_state.current_user
        st.title(f"Arbeitsverlauf für {current_user.firstname} {current_user.lastname}")

        # Aufgaben des aktuellen Nutzers abrufen
        task_data = load_task_data(current_user.id)

        # Wenn tasks gefunden
        if not task_data.empty:

            st.title("Statistik-Seite")


            st.subheader("Aufgabenübersicht")
            filtered_task_data = filter_dataframe(task_data)
            st.dataframe(filtered_task_data, hide_index=True)

            # Task-Auswahl
            selected_task_id = st.selectbox("Wähle eine Aufgabe", options=filtered_task_data['task_id'])

            if selected_task_id:
                st.subheader("Montageschritte der ausgewählten Aufgabe")
                task_steps_data = load_task_steps(selected_task_id)
                if not task_steps_data.empty:
                    st.dataframe(task_steps_data, hide_index=True)
                else:
                    st.write("Keine Montageschritte für diese Aufgabe gefunden.")

            # Durchschnittliche Zeit pro Aufgabe berechnen
            avg_time = task_data['Zeit'].mean()
            st.metric(label="Durchschnittliche Zeit pro Aufgabe", value=f"{avg_time:.2f} Sekunden")

            # Diagramm zur Visualisierung der Aufgaben
            st.subheader("Aufgabenverlauf")
            chart = alt.Chart(task_data).mark_bar().encode(
                x='Version',
                y='Zeit'
            )
            st.altair_chart(chart, use_container_width=True)

            # Export-Option
            st.subheader("Daten exportieren")
            csv = task_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="CSV-Datei herunterladen",
                data=csv,
                file_name='arbeitsverlauf.csv',
            )
        else:
            st.write("Keine Aufgaben gefunden")
    else:
        st.error("Bitte loggen Sie sich ein, um Ihren Arbeitsverlauf zu sehen.")
        #sleep(2)
        st.rerun


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


current_user = st.session_state.current_user

def task_details():

    # Lade alle Aufgaben aus der Datenbank
    task_data = load_task_data(current_user.id)

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

    # Anzeigen der Durchschnitllichen Zeit pro Version

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


def display_perceived_complexity_user(tasks_df: pd.DataFrame, versions_df: pd.DataFrame):
    st.subheader("Gefühlte Komplexität")
    users = session.query(User).all()

    if users:
        # Berechnen der gefühlten Komplexität für den Nutzer
        st.dataframe(average_perceived_complexity_per_version_for_user(tasks_df, versions_df, current_user.id), hide_index=True)


    else:
        st.warning("Es konnten keine Nutzer gefunden werden")


def display_box_plot_versions():
    # Daten aus der Datenbank abfragen
    tasks = session.query(Task.id, Task.time, Task.version_id, Version.name).join(Version).all()

    # DataFrame erstellen
    tasks_df = pd.DataFrame(tasks, columns=['task_id', 'time', 'version_id', 'version_name'])

    # Boxplot erstellen
    fig_completion_times = px.box(tasks_df, x='version_name', y='time',
                                  title='Verteilung der Bearbeitungszeiten nach Version')

    st.subheader("Box-Plots")

    # Diagramm anzeigen
    st.plotly_chart(fig_completion_times, use_container_width=True)



def display_learning_curve_user(tasks_with_versions: pd.DataFrame, users_df: pd.DataFrame):
    st.subheader("Lernkurve des Nutzers")

    # Versionsauswahl
    selected_version_learning_curve = st.selectbox("Wähle eine Version", versions_df['name'].unique(), index=None, placeholder="Wähle eine Version")

    # Wenn Nutzer und Version ausgewählt wurde
    if selected_version_learning_curve:

        selected_user_id = current_user.id

        if not tasks_with_versions.empty:

            user_data = tasks_with_versions[(tasks_with_versions['user_id'] == selected_user_id) &
                                            (tasks_with_versions['name'] == selected_version_learning_curve)]

            # Erstelle ein Liniendiagramm mit ausgewählten Daten
            fig = px.line(user_data, x=user_data.index, y=['time', 'perceived_complexity'],
                          title=f'Lernkurve für Nutzer "{current_user.firstname}{current_user.lastname}" und Version "{selected_version_learning_curve}"',
                          labels={'value': 'Wert', 'index': 'Aufgaben ID'},
                          line_shape='linear')

            # Füge die tatsächliche Komplexität als konstante Linie hinzu
            fig.add_scatter(x=user_data.index, y=user_data['complexity'], mode='lines', name='Tatsächliche Komplexität')

            # Anzeigen der IDs als category, sodass keine Gleitkommazahlen entstehen
            fig.update_xaxes(type='category')

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Es konnten keine Daten gefunden werden")



def display_histo():
    st.subheader('Häufigkeitsverteilung nach Version')

    df = load_task_data()

    if not df.empty:
        # Mittelwert und Standardabweichung berechnen
        mean = df['Zeit'].mean()
        stdev = df['Zeit'].std()

        if not np.isnan(stdev):
            stdev_plus = mean + stdev
            stdev_minus = mean - stdev

        # Min- und Max-Werte für die X-Achse definieren
        min_time = df['Zeit'].min()
        max_time = df['Zeit'].max()

        max_count = df['Zeit'].max()

        # Plotly Histogramm erstellen
        fig = px.histogram(
            df,
            x='Zeit',
            color='Version',
            title='Häufigkeitsverteilung der Prozesszeiten nach Versionen',
            labels={'Zeit': 'Prozesszeit', 'Version': 'Version'},
            nbins=30,  # Detailgrad der Gruppierungen
            #marginal='violin' # One of 'rug', 'box', 'violin', or 'histogram'
        )

        # Standardabweichung + Durchscnitt als Linie anzeigen
        fig.add_shape(type="line", x0=mean, x1=mean, y0=0, y1=1, xref='x', yref='y', line=dict(color='blue', dash='dash'), name='Durchschnitt', showlegend=True)


        if not np.isnan(stdev):

            fig.add_shape(type="line", x0=stdev_plus, x1=stdev_plus, y0=0, y1=1, xref='x', yref='y', line=dict(color='red', dash='dash'), name='Standardabweichung', showlegend=True)
            fig.add_shape(type="line", x0=stdev_minus, x1=stdev_minus, y0=0, y1=1, xref='x', yref='y', line=dict(color='red', dash='dash'), visible=True)

        # Layout des Histo anpassen
        fig.update_layout(
            xaxis_title='Prozesszeit',
            yaxis_title='Häufigkeit',
            # barmode='overlay',
            barmode='group',
            bargap=0.05,
            # bargroupgap=0.1
            xaxis=dict(range=[0, max_time])
        )

        st.plotly_chart(fig)

    else:
        st.warning("Es konnten keine Daten gefunden werden")


# Lade Daten in DataFrames
tasks_df = pd.read_sql(f'SELECT * FROM tasks WHERE user_id == {current_user.id}', engine)
versions_df = pd.read_sql('SELECT id, name, complexity FROM versions', engine)
users_df = pd.read_sql('SELECT id, firstname, lastname, nickname FROM users', engine)


# Verbinde tasks mit versions, um die Versionsnamen hinzuzufügen
tasks_with_versions = pd.merge(tasks_df, versions_df, left_on='version_id', right_on='id', suffixes=('', '_version'))

st.title("Statistik-Seite")

# Design der Seite
height_row_one = 700
height_row_two = 680

c1_1, c1_2 = st.columns([0.5, 0.5])
c2_1, c2_2 = st.columns(2)
c3_1, c3_2 = st.columns(2)
c4_1, c4_2 = st.columns(2)
c5_1, c5_2 = st.columns(2)

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
                display_histo()


        with c3_2:
            with st.container(border=True):
                display_learning_curve_user(tasks_with_versions, users_df)

    with st.container():
        with c4_1:
            with st.container(border=True):
                display_perceived_complexity_user(tasks_df, versions_df)

else:
    st.warning("Es konnten keine Daten gefunden werden. Bitte führen Sie zunächst einige Aufgaben im Arbeitsplatz-Bereich durch, um die Statistiken einzusehen!")


