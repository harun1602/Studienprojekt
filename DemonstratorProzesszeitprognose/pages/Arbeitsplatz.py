from pathlib import Path
import threading
from datetime import datetime
import streamlit as st
import pandas as pd
from data.database_code import TaskProfile, Version, ComponentList, ComponentListRequiredCount, TaskComponentRequirement
import time

from data.database_functions import load_images, save_task_steps, get_setting, save_task
from navigation import make_sidebar
from data.database_code import session

from streamlit.runtime.scriptrunner import add_script_run_ctx

import cv2
from Recognition.stack_interface import StackChecker


import sys
import subprocess
from multiprocessing.managers import BaseManager

PORT = 50055
AUTH = b"stackkey"

class ClientManager(BaseManager):
    pass


ClientManager.register("get_cmd_q")
ClientManager.register("get_status")

def connect_manager():
    m = ClientManager(address=("127.0.0.1", PORT), authkey=AUTH)
    m.connect()
    return m

def start_runner(variant="v1", camera="1"):
    proc = st.session_state.get("proc")
    if proc is not None and proc.poll() is None:
        st.warning("Runner läuft schon – starte keinen zweiten.")
    else:
        # Projekt-Root sauber bestimmen (ggf. parents[] anpassen!)
        ROOT = Path(__file__).resolve().parents[1]  
        RUNNER = ROOT / "Recognition" / "stack_runner.py"
        MODEL  = ROOT / "Recognition" / "best.pt"

        log_path = ROOT / "stack_runner.log"
        log_f = open(log_path, "a", encoding="utf-8")

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = subprocess.CREATE_NEW_CONSOLE

        st.session_state.proc = subprocess.Popen(
            [
                sys.executable, "-u", str(RUNNER),
                "--model", str(MODEL),
                "--variant", variant,
                "--camera", str(camera),
                "--port", str(PORT),
                "--auth", AUTH.decode("utf-8"),
            ],
            cwd=str(ROOT),                 # <<< wichtig
            stdout=log_f, stderr=log_f,    # <<< wichtig: Fehler landen im Log
            creationflags=creationflags,
        )
        time.sleep(0.6)
def beende_runner():
    try:
        m = connect_manager()
        m.get_cmd_q().put("stop")
    except Exception:
        pass

    proc = st.session_state.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass

    st.session_state.proc = None
    st.session_state.arbeitsplatz_stack_step_ready = False

def runner_nextStep():
    try:
        m = connect_manager()
        m.get_cmd_q().put("next")
    except Exception as e:
        st.warning(f"Runner nicht erreichbar: {e}")

def update_ready():
    try:
        m = connect_manager()
        status = m.get_status()
        # status ist ein Proxy-Dict, Werte direkt lesbar
        return dict(status)
    except Exception as e:
        st.warning(f"Runner nicht erreichbar: {e}")
        return {"ready": False, "running": False}

def arbeitsplatz_page():
    # Setzen des Seitenlayouts
    st.set_page_config(
        page_title="ProScheduler: Arbeitsplatz",
        initial_sidebar_state="expanded",
        layout="wide")

    # Einladen der Sidebar für die Navigation zwischen den Seiten
    make_sidebar()

    # Fixieren der Sidebar-Größe + Styleklassen für Timer + Countdown
    st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 250px;
           max-width: 250px;
       }
       [data-testid="sttextArea"][aria-expanded="true"]{
           min-width: 20px;
       }
       button[title="View fullscreen"]{
        visibility: hidden;}
       .time_out {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #ec5953 !important;
        }
        .time_in {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #228B22 !important;
        }
        </style> 
       """,
        unsafe_allow_html=True,
    )


    time_between_tasks = 0
    lapse = 0.1

    # Funktion für Timer-Thread
    def timer(timer_running):
        start_time = time.time()
        while timer_running.is_set():
            elapsed_time = time.time() - start_time
            st.session_state.time_elapsed = elapsed_time
            time.sleep(lapse)

    # Funktion für Countdown Thread
    def countdown(seconds, countdown_running):
        start_time = time.time()
        while countdown_running.is_set() and seconds > 0:
            elapsed_time = time.time() - start_time
            remaining_time = seconds - elapsed_time
            st.session_state.time_left = remaining_time
            time.sleep(lapse)

    # Der Nutzer hat die Möglichkeit seine aktuelle Session zurückzusetzen, um an weiteren Aufgaben zu arbeiten
    def reset_page():

        # resets = ['time_elapsed', 'timer_thread', 'timer_running', 'time_left', 'countdown_thread', 'countdown_running', 'timer_placeholder', 'selected_version', 'current_task',]
        resets = ['completed_tasks', 'remaining_tasks', 'current_profile', 'game_mode']
        # Delete all the items in Session state
        for key in resets:
            del st.session_state[key]

    # Dialog, um sicherzugehen, dass Session zurückgesetzt werden soll
    @st.dialog("Aufgaben zurücksetzen")
    def reset():
        st.write(
            "Sind Sie sicher, dass die Aufgaben zurückgesetzt werden sollen?  \n Die in der aktuellen Sitzung erledigten Aufgaben und die noch zu erledigenden Aufgaben werden zurückgesetzt.  \n\n **WICHTIG**: Die Datenbankeinträge bleiben weiterhin bestehen!")
        if st.button("Zurücksetzen"):
            reset_page()
            st.rerun()
    #Dialog, Wenn Modul nicht erkannt ist
    @st.dialog("Modul wurde nicht erkannt")
    def confirm_missing_module_dialog():
        st.warning("Modul wurde nicht erkannt. Wollen Sie trotzdem fortfahren?")

        col_yes, spacer, col_no = st.columns([1, 1.5, 1])
        with col_yes:
            if st.button("Ja, fortfahren",width="stretch"):
                st.session_state.confirm_missing_module = False
                st.session_state.force_advance = True
                st.rerun()

        with col_no:
            if st.button("Nein",width="stretch"):
                st.session_state.confirm_missing_module = False
                st.rerun()
    @st.dialog("Bauteil-Vorschau")
    def component_dialog():
        comp = st.session_state.selected_component
        if not comp:
            st.info("Kein Bauteil ausgewählt.")
            return

        st.write(f"**{comp['name']}**  \nAnzahl: **{comp['count']}**")

        from PIL import Image, ImageOps

        path = comp.get("image_path")
        if path:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)  # <-- korrigiert die EXIF-Drehung
            st.image(img, width="stretch")
        else:
            st.warning("Kein Bild für dieses Bauteil vorhanden.")


        spacer, col = st.columns([8, 3])
        with col:
            if st.button("Schließen", width="stretch"):
                st.session_state.show_component_dialog = False
                st.session_state.last_selected_component = None
                st.session_state.components_table_nonce += 1  # <-- Tabelle neu “mounten”
                st.rerun()


    # ----- Nachfolgenden werden die session-states definiert, welche für das korrekte beibehalten von Informationen/Variablenzuständen über Seiten-Aktualisierungen (also alle Operationen die einen refresh triggern -> bspw Button-Drücken) hinweg benötigt werden
    # Session State für Timer
    if 'time_elapsed' not in st.session_state:
        st.session_state.time_elapsed = 0.0

    if 'timer_thread' not in st.session_state:
        st.session_state.timer_thread = None

    if 'timer_running' not in st.session_state:
        st.session_state.timer_running = threading.Event()

    # Session State für countdown
    if 'time_left' not in st.session_state:
        st.session_state.time_left = 0.0

    if 'countdown_thread' not in st.session_state:
        st.session_state.countdown_thread = None

    if 'countdown_running' not in st.session_state:
        st.session_state.countdown_running = threading.Event()

    if 'timer_placeholder' not in st.session_state:
        st.session_state.timer_placeholder = ''

    # State zum Speichern der durch den Nutzer im Radio Button ausgewählten Version
    if 'selected_version' not in st.session_state:
        st.session_state.selected_version = None

    # State zum Speichern der aktuell laufenden Aufgabe
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
        st.session_state.start_time = None

    # State das noch eine Einschätzung des Nutzers nötig ist
    if 'task_just_ended' not in st.session_state:
        st.session_state.task_just_ended = False

    # State zum Speichern der aktuell laufenden Montageschrittes
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0

    # State zum Speichern der in der aktuellen Session erledigten Aufgabe
    if 'completed_tasks' not in st.session_state:
        st.session_state.completed_tasks = []

    # State zum Speichern der benötigten Bilder der aktuellen Aufgabe
    if 'image_paths' not in st.session_state:
        st.session_state.image_paths = []

    if 'image_instructions' not in st.session_state:
        st.session_state.image_instructions = []
    
    if 'images' not in st.session_state:
        st.session_state.image_instructions = []

    # State zum Speichern der benötigten Zeiten
    if 'steps_times' not in st.session_state:
        st.session_state.steps_times = []

    # State zum Speichern des aktuellen Spielmodus ('classic', 'timer', 'countdown')
    if 'game_mode' not in st.session_state:
        st.session_state.game_mode = ''

    # State zum Speichern, ob ein zufälliges Aufgabenprofil ausgewählt werden soll
    if 'random_profile' not in st.session_state:
        st.session_state.random_profile = False

    # State zum Speichern, ob ein zufälliger Spielmodus ausgewählt werden soll
    if 'random_mode' not in st.session_state:
        st.session_state.random_mode = False
    
    if "confirm_missing_module" not in st.session_state:
        st.session_state.confirm_missing_module = False

    if "force_advance" not in st.session_state:
        st.session_state.force_advance = False
    if "show_component_dialog" not in st.session_state:
        st.session_state.show_component_dialog = False
    if "selected_component" not in st.session_state:
        st.session_state.selected_component = None
    if "last_selected_component" not in st.session_state:
        st.session_state.last_selected_component = None
    if "components_table_nonce" not in st.session_state:
        st.session_state.components_table_nonce = 0

    if "timer_start_ts" not in st.session_state:
       st.session_state.timer_start_ts = None

    if "countdown_end_ts" not in st.session_state:
        st.session_state.countdown_end_ts = None

    # --- Stack / Livecam States ---
    if "stack_checker" not in st.session_state:
        st.session_state.stack_checker = None

    if "stack_cam_running" not in st.session_state:
        st.session_state.stack_cam_running = False

    if "arbeitsplatz_stack_step_ready" not in st.session_state:
        st.session_state.arbeitsplatz_stack_step_ready = False
    
    if "stack_variant" not in st.session_state:
        st.session_state.stack_variant = "v1"


    if "proc" not in st.session_state:
       st.session_state.proc = None    



    # Game_Modes
    modes = ["classic", "timer", "countdown"]

    # Wenn eingeloggt, dann öffne die Seite (Else fall kann eigentlich nicht eintreten, da mit Logout erneut auf die Anmelde-Seite geleitet)
    if 'logged_in' in st.session_state and st.session_state.logged_in:

        # Zufällige Auswahl des Aufgabenprofiles?
        random_profile = eval(get_setting('random_profile', "False"))

        # Zufällige Auswahl des Spielmodus?
        random_mode = eval(get_setting('random_mode', "False"))

        # Zufälliges Aufgabenprofil wählen
        if random_profile:

            if 'current_profile' not in st.session_state or st.session_state.current_profile is None:

                profiles = st.session_state.current_user.task_profiles

                if profiles:
                    st.session_state.current_profile = random.choice(profiles)
                else:
                    st.write("Keine Aufgabenprofile für den Nutzer vorhanden.")
                    return

        # Auswahl des Aufgabenprofils durch den Benutzer
        else:

            # Ziehe alle Profile des Nutzers
            profiles = st.session_state.current_user.task_profiles

            profiles = [profile for profile in profiles if profile.active]

            # Erstelle Namensliste der Profile, für das Auswahlfeld
            profile_names = [profile.name for profile in profiles]

            # Wenn Liste nicht leer
            if profile_names:
                selected_profile_name = st.selectbox("Wähle ein Aufgabenprofil aus", profile_names)
                if selected_profile_name and (('current_profile' not in st.session_state or st.session_state.current_profile is None) or (selected_profile_name != st.session_state.current_profile.name)):

                    st.session_state.current_profile = session.query(TaskProfile).filter_by(name=selected_profile_name).first()
                    st.session_state.completed_tasks = []

            # Wenn Liste leer
            else:
                st.write("Keine Aufgabenprofile für den Nutzer vorhanden.")

        if random_mode:
            # Prüfe, ob bereits ein Spielmodus festgelegt wurde
            if st.session_state.game_mode == '':
                st.session_state.game_mode = random.choice(modes)

        else:
            # Dem Nutzer angezeigte Auswahl
            mode_labels = ["Ohne Timer", "Mit Timer", "Countdown"]

            game_mode = st.selectbox("Wähle einen Modus", ["classic", "timer", "countdown"], index=0, format_func=lambda x: mode_labels[modes.index(x)], disabled=st.session_state.current_task is not None)

            if game_mode:
                st.session_state.game_mode = game_mode

        # Wenn Profil ausgewählt
        if 'current_profile' in st.session_state:

            # Speichern ausgewählten Profiles
            current_profile = st.session_state.current_profile


            # Verbleibende Aufgaben laden (Bei initial und Änderung des Aufgabenprofils)
            if 'remaining_tasks' not in st.session_state or current_profile.name not in st.session_state.remaining_tasks:
                remaining_tasks = {}

                if current_profile:
                    task_profile_name = current_profile.name
                    remaining_tasks[task_profile_name] = {}

                    # Für das jeweilige Task-Profil Auswahl speichern der benötigten Anzahl der Versionen
                    for required_count in current_profile.required_counts:
                        if required_count.version:
                            version_name = required_count.version.name
                            count = required_count.count
                            remaining_tasks[task_profile_name][version_name] = count

                # Speichern der Datenstruktur im session state
                st.session_state.remaining_tasks = remaining_tasks

            # ----- Seitenaufbau und Funktionalität

            # Ziehe den aktuell angemeldeten User
            current_user = st.session_state.current_user
            st.title(f"Arbeitsplatz für {current_user.firstname} {current_user.lastname} ({st.session_state.game_mode})")

            # Definieren der Spalten
            col1, col2, col3 = st.columns([0.25, 0.5, 0.25], gap='medium', vertical_alignment='top')

            # Aufgabenübersicht und Aufgabenauswahl
            with col1:
                with st.container(border=True):

                    # Ziehe der aktuell benötigten Aufgaben asu dem Session State
                    remaining_tasks = st.session_state.remaining_tasks

                    if remaining_tasks != {}:

                        # Übersicht über alle zu erledigenden Aufgaben
                        st.subheader("Zu erledigende Aufgaben")
                        st.write(f"Ausgewähltes Aufgabenprofil: {current_profile.name}")

                        # Radio-Button, um festzulegen, welche Version aktuell bearbeitet werden soll: Es werden nur alle Aufgaben mit Count > 0 angezeigt
                        filtered_versions = {k: v for k, v in st.session_state.remaining_tasks[current_profile.name].items() if v > 0}

                        # Wenn die gefilterten Aufgaben nicht leer sind
                        if filtered_versions:

                            st.session_state.task_running_bool = st.session_state.current_task is not None
                            st.session_state.selected_version = st.radio("Wähle eine Task-Version aus", list(filtered_versions.keys()), disabled=st.session_state.task_running_bool)
                            version = session.query(Version).filter_by(name=st.session_state.selected_version).first()

                            #print(f"version: {version.name}")

                            # Lade die Bilder für die Version
                            images = load_images(version.id)
                            image_paths = [image.image_path for image in images]
                            image_instructions = [image.image_anleitung for image in images]

                            # Speichere die image_paths im session stater
                            st.session_state.images = images
                            st.session_state.image_paths = image_paths
                            st.session_state.image_instructions = image_instructions
                            st.session_state.cover_image = version.cover_image_path

                            # Setzen des Countdowns auf initial Wert
                            if not st.session_state.task_just_ended and not st.session_state.countdown_running.is_set():
                                # Ermittle Zeitvorgabe für die Version
                                st.session_state.time_left = version.time_limit

                            # Setzen des Timers auf 0
                            if not st.session_state.task_just_ended and not st.session_state.timer_running.is_set():

                                st.session_state.time_elapsed = 0

                        # Ausgabe, dass alle Aufgaben des Aufgabenprofils bearbeitet sind
                        else:
                            st.warning("Gut gemacht! Alle Aufgaben des Aufgabenprofils wurden bearbeitet.")

                        # Iteriere über die Versionen im Task Profil und ihren aktuellen Count
                        for version_name, count in remaining_tasks[current_profile.name].items():
                            st.write(f"Version '{version_name}': {count} Mal")

                    # Wenn die Datenstruktur leer ist -> Fehler
                    else:
                        st.error("Beim laden der benötigten Aufgaben ist ein Fehler aufgetreten")
                
                with st.container(border=True):

                    # Erledigte Aufgaben anzeigen
                    st.subheader("Erledigte Aufgaben (aktuelle Sitzung)")
                    if 'completed_tasks' in st.session_state and st.session_state.completed_tasks != []:
                        for task in st.session_state.completed_tasks:
                            st.write(f"{task['version_name']}: {task['time']:.2f} Sekunden")
                    else:
                        st.write("Keine erledigten Aufgaben gefunden")
                

            # Anleitung und Start/Ende - Buttons
            with col2:

                # Wenn die Aufgabe gerade beendet wurde -> Feedback der Komplexität für den Nutzer anzeigen
                if st.session_state.task_just_ended:

                    with st.form("my_form"):

                        st.success("Sehr gut, Sie haben die Aufgabe beendet. Bitte geben Sie noch einige Einschätzungen zur Aufgabe ab.")

                        task = st.session_state.current_task

                        # Radio-Buttons für Nutzer-Einschätzungen
                        button_perceived_complexity = st.radio("Für wie schwer haben Sie die gesamte Aufgabe empfunden? (1 -> Einfach ; 4 -> Schwer)", [1, 2, 3, 4], index=None, horizontal=True)
                        button_perceived_stress = st.radio("Wie viel Stress haben Sie bei Erledigung der Aufgabe empfunden? (1 -> wenig ; 4 -> viel)", [1, 2, 3, 4], index=None, horizontal=True)
                        button_perceived_time_pressure = st.radio("Wie viel Zeitdruck haben Sie bei Erledigung der Aufgabe empfunden? (1 -> wenig ; 4 -> viel)", [1, 2, 3, 4], index=None, horizontal=True)
                        button_perceived_frustration = st.radio("Wie viel Frustration haben Sie bei Erledigung der Aufgabe empfunden? (1 -> wenig ; 4 -> viel)", [1, 2, 3, 4], index=None, horizontal=True)

                        #submitted = st.form_submit_button("Feedback geben", disabled= button_perceived_complexity == None or button_perceived_stress == None or button_perceived_time_pressure == None or button_perceived_frustration == None)

                        submitted = st.form_submit_button("Feedback geben")

                        # Nur, wenn alle Angaben getätigt wurden
                        if button_perceived_complexity and button_perceived_stress and button_perceived_time_pressure and button_perceived_frustration:
                            if submitted:
                                #print(f"Perceived_complexity: {button_perceived_complexity}")
                                task.perceived_complexity = button_perceived_complexity
                                task.perceived_stress = button_perceived_stress
                                task.perceived_time_pressure = button_perceived_time_pressure
                                task.perceived_frustration = button_perceived_frustration

                                print(f"Komplexität: {button_perceived_complexity}, Stress: {button_perceived_stress}, Zeitdruck: {button_perceived_time_pressure}, Frustration: {button_perceived_frustration}")

                                session.commit()

                                # Setze den Schritt zurück, wenn die Aufgabe abgeschlossen ist
                                st.session_state.current_step = 0
                                st.session_state.steps_times = []

                                # Zurücksetzen der Session States
                                st.session_state.current_task = None
                                st.session_state.start_time = None
                                st.session_state.selected_version = None
                                st.session_state.task_just_ended = False
                                st.rerun()

                else:
                    version = session.query(Version).filter_by(name=st.session_state.selected_version).first()
                    if version:
                        with st.container(border=True):

                            #st.subheader("Arbeitsumgebung")

                            # Zeige Cover-Bild, wenn keine Aufgabe läuft
                            if not st.session_state.current_task:
                                st.write(version.description)
                                if st.session_state.cover_image:
                                    st.image(st.session_state.cover_image, width='content', caption=f"Cover-Bild von {version.name}")
                                else:
                                    st.write("Kein Cover-Bild verfügbar.")
                            else:
                                # Zeige das aktuelle Bild für Montageschritt und Angabe des aktuellen Schrittes über Bild Caption
                                if st.session_state.current_step < len(st.session_state.image_paths):
                                    #test_anzeige = st.write(st.session_state.image_instructions[st.session_state.current_step], disabled=True)
                                    st.text_area('Anleitung', value=st.session_state.image_instructions[st.session_state.current_step], height=20, disabled=True)
                                    normalized_path = st.session_state.image_paths[st.session_state.current_step].replace("\\", "/")
                                    st.image(normalized_path,
                                             width="stretch"#,
                                             #caption=f"Schritt {st.session_state.current_step + 1}/{len(st.session_state.image_paths)}"
                                             )
                                else:
                                    st.write("Alle Montageschritte abgeschlossen.")

                            # Start- und Stop-Buttons nebeneinander anzeigen
                            col2_1, col2_2, col2_3 = st.columns([0.25, 0.5, 0.25])
                            col2_2_1, col2_2_2 = st.columns([1,1])
                            with col2_1:

                                with col2_2_1:
                                    status_placeholder = st.empty()

                                if 'current_task' in st.session_state and st.session_state.current_task is not None:

                                    # Ausgabe, dass der Timer läuft
                                    status_placeholder.success(f"Task '{version.name}' gestartet! Timer läuft...")

                                # Wenn "Starte Aufgabe" - Button gedrückt wird
                                if st.button("Starte Aufgabe", disabled=True if st.session_state.current_task else False):

                                    # Nur, wenn der "Starte Aufgabe" - Button nicht bereits vorher gedrückt wurde
                                    if 'current_task' not in st.session_state or st.session_state.current_task is None:
                                        # Falls Spielmodus 'Timer'
                                        if st.session_state.game_mode == 'timer':

                                            # Starte Timer
                                            if st.session_state.timer_thread is None or not st.session_state.timer_thread.is_alive():
                                                # st.session_state.timer_running.set()  # Timer starten
                                                # st.session_state.timer_thread = threading.Thread(target=timer, args=(
                                                #     st.session_state.timer_running,))
                                                # add_script_run_ctx(st.session_state.timer_thread)
                                                # st.session_state.timer_thread.start()
                                                st.session_state.timer_running.set()
                                                st.session_state.timer_start_ts = time.time()   

                                        # Falls Spielmodus 'countdown'
                                        if st.session_state.game_mode == 'countdown':

                                            # Starte Countdown
                                            if st.session_state.countdown_thread is None or not st.session_state.countdown_thread.is_alive():
                                                # st.session_state.countdown_running.set()  # Countdown starten
                                                # version = session.query(Version).filter_by(
                                                #     name=st.session_state.selected_version).first()
                                                # st.session_state.countdown_thread = threading.Thread(target=countdown, args=(version.time_limit,
                                                #     st.session_state.countdown_running,))
                                                # add_script_run_ctx(st.session_state.countdown_thread)
                                                # st.session_state.countdown_thread.start()
                                                st.session_state.countdown_running.set()
                                                st.session_state.countdown_end_ts = time.time() + version.time_limit
                                                st.session_state.time_left = version.time_limit

                                        # Ziehen der aktuellen Version
                                        version = session.query(Version).filter_by(name=st.session_state.selected_version).first()
                                        if version:
                                            # Starte Timer für die gesamte Aufgabe
                                            st.session_state.start_time = datetime.now()
                                            # Starte Timer für den ersten Montageschritt
                                            st.session_state.steps_times.append({'start_time': datetime.now()})

                                            # Erstellen einer neuen Task in der Datenbank
                                            #new_task = Task(user_id=current_user.id, version_id=version.id, start_timestamp=st.session_state.start_time, time=0, game_mode=st.session_state.game_mode)
                                            #session.add(new_task)
                                            #session.commit()

                                            new_task = save_task(user_id=current_user.id, version_id=version.id, start_timestamp=st.session_state.start_time, time=0, game_mode=st.session_state.game_mode)

                                            st.session_state.current_task = new_task

                                            # start_stack_cam()
                                            # if st.session_state.proc is None or st.session_state.proc.poll() is not None:
                                            #     # NEW_CONSOLE -> separates Konsolenfenster (optional, OpenCV Fenster kommt sowieso)
                                            #     creationflags = 0
                                            #     if sys.platform.startswith("win"):
                                            #         creationflags = subprocess.CREATE_NEW_CONSOLE

                                            #     st.session_state.proc = subprocess.Popen(
                                            #         [sys.executable, "stack_runner.py", "--model", "best.pt", "--variant", "v1",
                                            #         "--camera", "1", "--port", str(PORT), "--auth", AUTH.decode("utf-8")],
                                            #         creationflags=creationflags
                                            #     )
                                            #     time.sleep(0.6)  # kurz warten, bis IPC-Server da ist
                                            start_runner()

                                            with col2_2_1:
                                                # Ausgabe, dass der Timer läuft
                                                st.success(f"Task '{version.name}' gestartet! Timer läuft...")

                                            st.rerun()
                                        else:
                                            st.error("Version nicht gefunden.")

                                    # Sollte der Timer bereits laufen, gib Hinweis aus
                                    else:
                                        status_placeholder.warning("Die Aufgabe wurde bereits gestartet")
                                        time.sleep(time_between_tasks)
                                        status_placeholder.success("Timer läuft...")

                            with col2_3:
                                def advance_step_or_finish():
                                    # Wenn es sich nicht um den letzten Montage-Schritt handelt
                                    if st.session_state.current_step < len(st.session_state.image_paths) - 1:

                                        with col2_2_2:
                                            st.success("Montageschritt abgeschlossen. Bitte beginn unverzüglich mit dem nächsten Schritt. Der timer läuft weiter")

                                        # Erfasse Endzeit für den aktuellen Schritt
                                        end_time = datetime.now()
                                        if st.session_state.current_step >= 0:
                                            previous_step = st.session_state.steps_times[len(st.session_state.steps_times) - 1]
                                            previous_step["end_time"] = end_time
                                            previous_step["time_spent"] = (end_time - previous_step["start_time"]).total_seconds()

                                        # Starte den nächsten Schritt
                                        st.session_state.steps_times.append({"start_time": end_time})
                                        st.session_state.current_step += 1

                                        # >>> StackChecker Step mitziehen
                                        runner_nextStep()


                                        time.sleep(time_between_tasks)
                                        st.rerun()

                                    # Wenn es sich um den letzten Montageschritt handelt
                                    else:

                                        if st.session_state.game_mode == "timer":
                                            st.session_state.timer_running.clear()  # Timer stoppen
                                            

                                        if st.session_state.game_mode == "countdown":
                                            st.session_state.countdown_running.clear()  # Countdown stoppen

                                        # Ende Zeit der gesamten Aufgabe bestimmen + benötigte Zeit
                                        end_time = datetime.now()
                                        elapsed_time = (end_time - st.session_state.start_time).total_seconds()
                                        task = st.session_state.current_task
                                        task.end_timestamp = end_time
                                        task.time = elapsed_time
                                        session.commit()

                                        # Letzter Eintrag wird gefüllt
                                        st.session_state.steps_times[st.session_state.current_step]["end_time"] = end_time
                                        st.session_state.steps_times[st.session_state.current_step]["time_spent"] = (
                                            end_time - st.session_state.steps_times[st.session_state.current_step]["start_time"]
                                        ).total_seconds()

                                        # speichere die Montage-Schritte der aktuellen Aufgabe
                                        save_task_steps(task.id, st.session_state.steps_times)

                                        # Einfügen der aktuellen Aufgabe zu den in der Session erledigten Aufgaben
                                        st.session_state.completed_tasks.append({
                                            "version_name": task.version.name,
                                            "time": task.time,
                                            "start_timestamp": task.start_timestamp,
                                            "end_timestamp": task.end_timestamp
                                        })

                                        # Aktualisieren der remaining_tasks
                                        task_profile_name = list(st.session_state.remaining_tasks.keys())[0]
                                        if st.session_state.remaining_tasks[task_profile_name][st.session_state.selected_version] > 0:
                                            st.session_state.remaining_tasks[task_profile_name][st.session_state.selected_version] -= 1

                                        # Ausgabe der Zeit für jeden Schritt
                                        for idx, step in enumerate(st.session_state.steps_times):
                                            print(f"Schritt {idx + 1}: {step['time_spent']:.2f} Sekunden")

                                        with col2_2_2:
                                            st.success("Aufgabe abgeschlossen, Gut gemacht!")
                                        
                                        #Aufgabe beenden
                                        beende_runner()
                                        st.session_state.task_just_ended = True
                                        st.rerun()
                               
                                if st.session_state.force_advance:
                                    st.session_state.force_advance = False
                                    advance_step_or_finish()
                                # Nutzer drückt Beenden Knopf
                                if st.button(label="Beende Schritt" if st.session_state.current_step < len(st.session_state.image_paths)-1 else "Beende Aufgabe", disabled=False if st.session_state.current_task else True):
                                    
                                    status = update_ready()
                                    erkannt = bool(status.get("ready", False))


                                    # Wenn aktuell eine Aufgabe läuft
                                    if st.session_state.current_task:
                                        
                                        
                                        if erkannt:
                                            advance_step_or_finish()
                                            
                                        else:
                                            # Dialog öffnen
                                            confirm_missing_module_dialog()
                                    # Ausgabe, dass keine laufende Task gefunden wurde -> Keine Aufgabe wurde bisher gestartet
                                    else:
                                        st.error("Keine laufende Task gefunden.")

                                # if st.session_state.confirm_missing_module:
                                #     confirm_missing_module_dialog() 
                    # Ausgabe, dass keine weiteren Aufgaben zu erledigen sind
                    else:
                        st.success("Keine Aufgabe mehr zu erledigen für das aktuelle Aufgabenprofil! Gut Gemacht!")

            # Fortschrittsanzeige
            with (col3):
                # livecam
                with st.container(border=True):
                    ROOT = Path(__file__).resolve().parents[1]
                    log_path = ROOT / "stack_runner.log"

                    if st.button("Runner-Log anzeigen"):
                        if log_path.exists():
                            st.code(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
                        else:
                            st.info("Noch kein Log vorhanden.")


                if st.session_state.game_mode != 'classic':

                    with st.container(border=True):
                        timer_ph = st.empty()

                        @st.fragment(run_every=0.2)
                        def render_timer():
                            if st.session_state.game_mode == "timer":
                                if st.session_state.timer_running.is_set() and st.session_state.timer_start_ts is not None:
                                    st.session_state.time_elapsed = time.time() - st.session_state.timer_start_ts

                                timer_ph.markdown(
                                    f"""
                                    <p class="time_in">
                                        Vergangene Zeit:<br />{st.session_state.time_elapsed:.1f} Sekunden
                                    </p>
                                    """,
                                    unsafe_allow_html=True
                                )

                            elif st.session_state.game_mode == "countdown":
                                if st.session_state.countdown_running.is_set() and st.session_state.countdown_end_ts is not None:
                                    st.session_state.time_left = st.session_state.countdown_end_ts - time.time()

                                time_left = float(st.session_state.time_left)
                                time_class = "time_in" if time_left >= 0 else "time_out"

                                timer_ph.markdown(
                                    f"""
                                    <p class="{time_class}">
                                        Verbleibende Zeit für Fertigung:<br />{time_left:.1f} Sekunden
                                    </p>
                                    """,
                                    unsafe_allow_html=True
                                )
                        render_timer()

                if st.session_state.current_task is None:

                    with st.container(border=True):

                        version = session.query(Version).filter_by(name=st.session_state.selected_version).first()

                        if version:
                            matching_component_list = session.query(ComponentList).filter_by(version_id=version.id).first()

                            if matching_component_list:
                                st.write(f"Bauteile für die Version: {st.session_state.selected_version}; Liste: {matching_component_list.name}")

                                # Bauteile mit ihren Namen und Counts anzeigen
                                required_counts = matching_component_list.required_counts

                                # Erstelle eine Liste der Bauteile und ihrer Count-Werte
                                component_data = [(rc.component.name, rc.count) for rc in required_counts]

                                # Konvertiere die Liste in ein DataFrame
                                df = pd.DataFrame(component_data, columns=["Bauteil", "Anzahl"])

                                # Zeige die Tabelle ohne Index an
                                st.dataframe(df, hide_index=True)

                            else:
                                st.write("Keine zugewiesene Bauteilliste für diese Version gefunden.")
                        else:
                            st.write("Probleme mit auslesen der Version ")
                    
                    if st.button("Reset"):
                        reset()

                if st.session_state.current_task is not None and not st.session_state.task_just_ended:

                    with st.container(border=True):
                        
                        # Ziehen der Version-Informationen
                        version = session.query(Version).filter_by(name=st.session_state.selected_version).first()

                        if version:
                            if st.session_state.current_step < len(st.session_state.image_paths):

                                st.write(f"Bauteile für den Schritt {st.session_state.current_step}")

                                # Ziehe die aktuellen Schritt-Informationen über das Image des Schrittes
                                current_image = version.images[st.session_state.current_step]

                                # Ziehe benötigte Anzahl der Bauteile für diesen Schritt
                                required_counts = session.query(TaskComponentRequirement).filter_by(image_id=current_image.id).all()

                                rows = []
                                for rc in required_counts:
                                    rows.append({
                                        "Bauteil": rc.component.name,
                                        "Anzahl": rc.count,
                                        "_img": getattr(rc.component, "component_image_path", None)  # <-- ggf. Feldnamen anpassen
                                    })

                                df = pd.DataFrame(rows)

                                if df.empty:
                                    st.warning("Keine Bauteile für diesen Schritt angegeben.")
                                else:
                                    event = st.dataframe(
                                        df[["Bauteil", "Anzahl"]],
                                        use_container_width=True,   # statt width="stretch"
                                        hide_index=True,
                                        key=f"components_step_{st.session_state.current_step}_{st.session_state.components_table_nonce}",
                                        selection_mode="single-cell",
                                        on_select="rerun",
                                    )

                                    row_idx = None
                                    if hasattr(event, "selection") and getattr(event.selection, "cells", None):
                                        row_idx = event.selection.cells[0][0]

                                    if row_idx is not None:
                                        name = df.loc[row_idx, "Bauteil"]
                                        st.session_state.last_selected_component = name
                                        st.session_state.selected_component = {
                                            "name": name,
                                            "count": int(df.loc[row_idx, "Anzahl"]),
                                            "image_path": df.loc[row_idx, "_img"],
                                        }
                                        st.session_state.show_component_dialog = True

                                    if st.session_state.show_component_dialog:
                                        component_dialog()

                            else:
                                st.warning("Keine Bauteile für diesen Schritt angegeben.")


        # Ausgabe, wenn kein Aufgabenprofil in der Datenbank hinterlegt ist
        else:
            st.error("Es konnte kein Aufgabenprofil gefunden. Bitte deinen Admin eines anzulegen.")

    else:
        st.error("Bitte loggen Sie sich ein, um Ihren Arbeitsplatz zu sehen.")


# Aufrufen der Seite
arbeitsplatz_page()
