import os
import shutil
import time
import streamlit as st
from sqlalchemy import event
from data.database_code import session, User, Component, ComponentList, ComponentListRequiredCount, TaskProfile, Version, TaskProfileRequiredCount, Image, TaskComponentRequirement
from data.database_functions import get_task_profiles, get_versions, \
    update_task_profile, load_task_data, delete_filtered_tasks, add_task_profile_to_users, toggle_version_status, \
    toggle_task_profile_status, get_setting, save_setting, is_version_unique, create_new_version, update_version, \
    is_component_unique, create_new_component, delete_all_components, delete_component, update_component, check_component_discrepancies
from data.filter_dataframe import filter_dataframe
from navigation import make_sidebar


st.set_page_config(initial_sidebar_state="expanded")

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

def clear_input():
    st.session_state.version_name = ""  # Leert das Text Input Feld

    st.session_state.name_input = None
    st.session_state.complexity_input = None
    st.session_state.time_limit_input = None
    st.session_state.selected_task_profiles = None
    #st.session_state.task_profiles_input.value = None
    st.session_state.new_version_description = None
    st.session_state.component_image = None


# Benötigte session_states

if 'clear_inputs' not in st.session_state:
    st.session_state.clear_inputs = False

if st.session_state.clear_inputs:
    clear_input()
    st.session_state.clear_inputs = False

# Session State zum Leeren des File-Uploads
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "update_lists" not in st.session_state:
    st.session_state.update_lists = True

# lokales Speichern der Bilder mit revision nummer
def save_images_with_revision(revision_number, sorted_images, image_texts, image_directory):

    revision_folder = os.path.join(image_directory, f"revision_{revision_number}")

    # Falls der Ordner nicht existiert, erstellen
    if not os.path.exists(revision_folder):
        os.makedirs(revision_folder)

    images = []

    for uploaded_file in sorted_images:

        # Dateipfad für das Bild definieren, basierend auf der Revision
        image_path = os.path.join(revision_folder, uploaded_file.name)

        # Speichern der Datei
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Bild '{uploaded_file.name}' erfolgreich hochgeladen und in Revision {revision_number} gespeichert.")

        # Bildinformationen in images einfügen
        image_info = {
            'image_path': image_path,
            'image_anleitung': image_texts.get(f'image_{uploaded_file.name}')
        }
        images.append(image_info)

    return images


def delete_images_of_version(version_id):
    # Abfrage der Version mit den zugehörigen Bildern
    version = session.query(Version).filter(Version.id == version_id).first()

    if version:
        # Iteriere über die Bilder der Version und lösche sie
        for image in version.images:
            session.delete(image)

        # Änderungen übernehmen
        session.commit()
    else:
        print("Version nicht gefunden")


# Löschen der Bilder im Ordner der gelöschten Version
def delete_image_folder(images):
    if images:
        # Nehme den Pfad des ersten Bildes und gehe davon aus, dass alle Bilder im selben Ordner sind
        folder_path = os.path.dirname(images[0].image_path)
        try:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # Überprüfen, ob der Ordner tatsächlich Bilder enthält
                if any(os.path.isfile(os.path.join(folder_path, f)) for f in os.listdir(folder_path)):
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
                else:
                    print(f"Folder is empty or does not contain files: {folder_path}")
            else:
                print(f"Folder not found or not a directory: {folder_path}")
        except Exception as e:
            print(f"Error deleting folder {folder_path}: {e}")


def version_creation_2():
    with st.container(border=True):

        task_profiles = get_task_profiles()
        task_profile_dict = {tp.id: tp for tp in task_profiles}

        st.subheader("Neue Version hinzufügen")


        # Eingabefelder für Standardwerte
        name = st.text_input("Name der Version",value= None, key='name_input')
        complexity = st.number_input("Komplexität", min_value=1.0, max_value=10.0, step=0.1, value=None, key='complexity_input')
        time_limit = st.number_input("Vorgegebene Zeit (Sekunden)", min_value=1, max_value=1000, step=1, value=None, key='time_limit_input')

        # Mehrfachauswahl für TaskProfiles
        selected_task_profiles = st.multiselect(
            "Welchen Taskprofiles soll diese Version zugewiesen werden? (optional)",
            [tp.id for tp in task_profiles],
            format_func=lambda x: task_profile_dict[x].name,
            key='task_profiles_input',
        )

        task_profile_count = {}

        # Für jedes ausgewählte TaskProfil muss die Anzahl der zu erledigenden Aufgaben festgelegt werden
        for tp_id in selected_task_profiles:
            task_profile_count[tp_id] = st.number_input(f"Anzahl für {task_profile_dict[tp_id].name}", min_value=1, value=1, key=f"count_{task_profile_dict[tp_id].name}_version_creation")

        # Bool - Wert zum Prüfen der Eingaben
        all_inputs = name and complexity and time_limit

        if all_inputs:

            with st.container(border=True):

                st.write("**Hier Coverbild hochladen**", unsafe_allow_html=True)

                cover_upload = st.file_uploader("Coverbild hochladen", accept_multiple_files=False)

                if cover_upload:
                    bytes_data = cover_upload.read()
                    st.image(bytes_data)

                version_beschreibung_input = st.text_input(
                    "Versionsbeschreibung 👇",
                    placeholder='Beschreibe hier KURZ  die Version',
                )

            with st.container(border=True):

                images_upload = st.file_uploader("Anleitungsbilder hochladen", accept_multiple_files=True)
                # Erstelle eine leere Liste, um die Texte zu speichern
                image_texts = {}

                if images_upload:
                    # Sortiere die hochgeladenen Dateien nach ihrem Namen
                    sorted_images = sorted(images_upload, key=lambda x: x.name)

                    for uploaded_file in sorted_images:
                        bytes_data = uploaded_file.read()
                        st.image(bytes_data, use_column_width=True)

                        # Erzeuge ein eindeutigen Key für jedes Bild
                        text_key = f'image_{uploaded_file.name}'

                        # Erstelle ein Textinput-Feld mit dem spezifischen Key
                        image_text = st.text_area(
                            "Anleitungstext zum Bild 👇",
                            placeholder=f'Beschreibe hier die Arbeitsschritte zum Bild {uploaded_file.name}',
                            key=text_key
                        )

                        # Speichere den eingegebenen Text in der Liste unter dem jeweiligen Key
                        image_texts[text_key] = image_text

            # Prüfe, ob Cover hochgeladen und Versionsbeschreibung vorgenommen
            cover_filled = cover_upload and version_beschreibung_input

            # Prüfe, ob alle Versionen ausgefüllt sind
            all_versions_filled = all(value.strip() for value in image_texts.values()) and images_upload

            if st.button("Version erstellen"):

                if all_inputs and all_versions_filled and cover_filled and is_version_unique(name):

                    # Verzeichnis für Anleitungsbilder erstellen, falls es nicht existiert
                    image_directory = f'images/{name}'
                    if not os.path.exists(image_directory):
                        os.makedirs(image_directory)

                    # hochladen des Cover-Bildes
                    image_path = os.path.join(image_directory, f"cover_{cover_upload.name}")
                    with open(image_path, "wb") as f:
                        f.write(cover_upload.getbuffer())
                    st.success(f"Cover-Bild '{cover_upload.name}' erfolgreich hochgeladen und gespeichert.")

                    # Lokales Speichern der angegebenen Bilder
                    images = save_images_with_revision(1,sorted_images, image_texts, image_directory)

                    # Erstelle neue Version
                    new_version = create_new_version(name, version_beschreibung_input, complexity, f'{image_directory}/cover_{cover_upload.name}', time_limit, images)

                    # Speichern der neu zugewiesenen TaskProfiles
                    for tp_id in selected_task_profiles:

                        task_profile = task_profile_dict[tp_id]

                        # Neuen Eintrag in TaskProfileRequiredCount erstellen
                        new_count = TaskProfileRequiredCount(
                            task_profile_id=task_profile.id,
                            version_id=new_version.id,
                            count=task_profile_count[tp_id]
                        )
                        session.add(new_count)

                        # Die neue Version zu den TaskProfiles hinzufügen
                        task_profile.versions.append(new_version)

                    session.commit()

                    st.success("Neue Version hinzugefügt!")

                    # Warte 2 Sekunden und Lade dann die Seite neu
                    time.sleep(2)

                    st.session_state.clear_inputs = True


                    # Seite neu laden, damit die neue Version auch sofort in der Erstellung neuer Aufgabenprofile genutzt werden kann
                    st.rerun()

                else:

                    if not is_version_unique(name):
                        st.toast("**Der Versionsname ist bereits vorhanden. Bitte wählen Sie einen anderen Namen**", icon="⚠️")
                    else:
                        st.toast("**Bitte Fülle zunächst alle Felder aus, bevor du eine Version erstellst**", icon="⚠️")


def version_edit_2():
    st.subheader("Version bearbeiten")
    # ziehen der Versionen
    versions = session.query(Version).all()

    # Liste über alle Versionsnamen
    version_names = [version.name for version in versions]

    # Selectbox zum Auswählen der gewünscten Version
    selected_version_name = st.selectbox("Wählen Sie eine Version zur Bearbeitung", version_names)

    if selected_version_name:

        # Ziehen der ausgewählten Version
        selected_version = session.query(Version).filter_by(name=selected_version_name).first()
        if selected_version:

            with st.container(border=True):

                # Standardwerte schreiben
                new_name = st.text_input("Name der Version", value=selected_version.name)
                new_complexity = st.number_input("Komplexität", min_value=1.0, max_value=10.0, step=0.1, value=selected_version.complexity)
                new_time_limit = st.number_input("Vorgegebene Zeit (Sekunden)", min_value=1, max_value=1000, step=1, value=selected_version.time_limit)

                all_inputs = new_name and new_complexity and new_time_limit

                if all_inputs:

                    with st.container(border=True):

                        st.write("**Hier Coverbild hochladen**", unsafe_allow_html=True)

                        new_cover_upload = st.file_uploader("Neues Coverbild hochladen", accept_multiple_files=False, key=f"cover_uploader_{st.session_state.uploader_key}")

                        if new_cover_upload:
                            bytes_data = new_cover_upload.read()
                            st.image(bytes_data)

                        new_version_beschreibung_input = st.text_input(
                            "Neue Versionsbeschreibung 👇",
                            placeholder='Beschreibe hier KURZ die Version',
                            key ="new_version_description"
                        )

                    with st.container(border=True):

                        new_images_upload = st.file_uploader("Neue Anleitungsbilder hochladen", accept_multiple_files=True, key=f"image_uploader_{st.session_state.uploader_key}" )
                        # Erstelle eine leere Liste, um die Texte zu speichern
                        new_image_texts = {}

                        if new_images_upload:
                            # Sortiere die hochgeladenen Dateien nach ihrem Namen
                            sorted_images = sorted(new_images_upload, key=lambda x: x.name)

                            for uploaded_file in sorted_images:
                                bytes_data = uploaded_file.read()
                                st.image(bytes_data, use_column_width=True)

                                # Erzeuge einen eindeutigen Key für jedes Bild
                                text_key = f'image_{uploaded_file.name}'

                                # Erstelle ein Textinput-Feld mit dem spezifischen Key
                                image_text = st.text_area(
                                    "Anleitungstext zum Bild 👇",
                                    placeholder=f'Beschreibe hier die Arbeitsschritte zum Bild {uploaded_file.name}',
                                    key=text_key
                                )

                                # Speichere den eingegebenen Text in der Liste unter dem jeweiligen Key
                                new_image_texts[text_key] = image_text

                    # Prüfe, ob Cover hochgeladen
                    cover_filled = new_cover_upload and new_version_beschreibung_input

                    # Prüfe, ob alle Versionen ausgefüllt sind
                    all_versions_filled = all(value.strip() for value in new_image_texts.values())

                    if st.button("Version speichern"):
                        if all_inputs and all_versions_filled:

                            # Verzeichnis für Anleitungsbilder erstellen, falls es nicht existiert
                            image_directory = f'images/{new_name}'
                            if not os.path.exists(image_directory):
                                os.makedirs(image_directory)

                            # Cover-Bild hochladen
                            new_cover_image_path = None
                            if cover_filled:
                                new_cover_image_path = os.path.join(image_directory, f"cover_{new_cover_upload.name}")
                                with open(new_cover_image_path, "wb") as f:
                                    f.write(new_cover_upload.getbuffer())
                                st.success(f"Cover-Bild '{new_cover_upload.name}' erfolgreich hochgeladen und gespeichert.")

                            # Verarbeiten der neuen Bilder
                            updated_images = []
                            if new_images_upload:

                                # Neues Bild hochladen
                                for uploaded_file in new_images_upload:
                                    #lokales speicher des Bildes
                                    image_path = os.path.join(image_directory, uploaded_file.name)
                                    with open(image_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    st.success(f"'{uploaded_file.name}' erfolgreich hochgeladen und gespeichert.")

                                    # Füge das Bild der Liste hinzu, die an update_version übergeben wird
                                    updated_images.append({
                                        'image_path': image_path,
                                        'image_anleitung': new_image_texts.get(f'image_{uploaded_file.name}')
                                    })

                            # Aufruf von update_version
                            update_version(
                                version_id=selected_version.id,
                                new_name=new_name,
                                new_description=new_version_beschreibung_input,
                                new_complexity=new_complexity,
                                new_cover_image_path=new_cover_image_path,
                                new_time_limit=new_time_limit,
                                updated_images=updated_images
                            )

                            st.session_state.clear_inputs = True
                            st.session_state.uploader_key += 1

                            st.success(f"Version '{new_name}' erfolgreich gespeichert!")
                            time.sleep(2)
                            st.rerun()


                        else:
                            st.toast("**Bitte Fülle zunächst alle Felder aus, bevor du eine Version erstellst**", icon="⚠️")


def version_edit():

    st.subheader("Version bearbeiten")
    # ziehen der Versionen
    versions = session.query(Version).all()

    # Liste über alle Versionsnamen
    version_names = [version.name for version in versions]

    # Selectbox zum Auswählen der gewünschten Version
    selected_version_name = st.selectbox("Wählen Sie eine Version zur Bearbeitung", version_names)

    if selected_version_name:

        # Ziehen der ausgewählten Version
        selected_version = session.query(Version).filter_by(name=selected_version_name).first()
        if selected_version:

            # Angaben der Versionswerte als "Standartwerte"
            new_name = st.text_input("Name der Version", value=selected_version.name)
            new_instructions = st.text_area("Anleitung", value=selected_version.instructions)
            new_complexity = st.number_input("Komplexität", min_value=1.0, max_value=10.0, step=0.1, value=selected_version.complexity)
            new_time_limit = st.number_input("Vorgegebene Zeit (Sekunden)", min_value=1, max_value=1000, step=1, value=selected_version.time_limit, key='new_time_limit')

            # Hier können ggf Bilder für die Anleitungen der Versionen hochgeladen werden
            new_cover_image_upload = st.file_uploader("Coverbild hochladen", accept_multiple_files=False, key=f"cover_uploader_edit")

            # Hier muss ggf noch angepasst werden: Anzeige der aktuellen Bilder + diese ggf löschen
            uploaded_files = st.file_uploader("Bilder hochladen (optional)", accept_multiple_files=True)

            if st.button("Version speichern"):

                # Nur wenn alle Angaben getätigt
                if new_name and new_instructions and new_complexity and new_time_limit and new_cover_image_upload:

                    # Hochladen der Bilder + Anhängen an Version: Hier fehlt noch ggf. löschen der Alten
                    if uploaded_files or new_cover_image_upload:
                        image_directory = f'images/{new_name}'
                        if not os.path.exists(image_directory):
                            os.makedirs(image_directory)

                    if new_cover_image_upload:
                        new_cover_image_path = os.path.join(image_directory, f"cover_{new_cover_image_upload.name}")
                        with open(new_cover_image_path, "wb") as f:
                            f.write(new_cover_image_upload.getbuffer())
                        st.success(f"Cover-Bild '{new_cover_image_upload.name}' erfolgreich hochgeladen und gespeichert.")

                    # Werte anpassen
                    selected_version.name = new_name
                    selected_version.instructions = new_instructions
                    selected_version.complexity = new_complexity
                    selected_version.cover_image_path = new_cover_image_path
                    selected_version.time_limit = new_time_limit
                    session.commit()

                    # Hochladen der Bilder + Anhängen an Version: Hier fehlt noch ggf. löschen der Alten
                    if uploaded_files:
                        image_directory = f'images/{new_name}'
                        if not os.path.exists(image_directory):
                            os.makedirs(image_directory)

                        for uploaded_file in uploaded_files:
                            image_path = os.path.join(image_directory, uploaded_file.name)
                            with open(image_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            st.success(f"Cover-Bild '{uploaded_file.name}' erfolgreich hochgeladen und gespeichert.")
                            new_image = Image(version_id=selected_version.id, image_path=image_path)
                            session.add(new_image)
                        session.commit()

                    st.success(f"Version '{new_name}' erfolgreich gespeichert!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Bitte füllen Sie alle erforderlichen Felder aus.")


def version_creation():

    # Neue Version hinzufügen - Es werden die Attribute Name, instructions und complexity benötigt
    st.subheader("Neue Version hinzufügen")
    name = st.text_input("Name der Version")
    instructions = st.text_area("Anleitung")
    complexity = st.number_input("Komplexität", min_value=1.0, max_value=10.0, step=0.1)
    time_limit = st.number_input("Vorgegebene Zeit (Sekunden)", min_value=1, max_value=1000, step=1)

    # Hier können ggf Bilder für die Anleitungen der Versionen hochgeladen werden
    cover_image_upload = st.file_uploader("Coverbild hochladen", accept_multiple_files=False)

    # Hier können ggf Bilder für die Anleitungen der Versionen hochgeladen werden
    uploaded_files = st.file_uploader("Anleitungsbilder hochladen", accept_multiple_files=True, key=f"cover_uploader_creation")

    # Falls Version neu hinzugefügt wird
    if st.button("Version hinzufügen"):

        if name and instructions and complexity and uploaded_files and time_limit and cover_image_upload:

            if uploaded_files or cover_image_upload:
                # Verzeichnis für Anleitungsbilder erstellen, falls es nicht existiert
                image_directory = f'images/{name}'
                if not os.path.exists(image_directory):
                    os.makedirs(image_directory)

            if cover_image_upload:
                image_path = os.path.join(image_directory, f"cover_{cover_image_upload.name}")
                with open(image_path, "wb") as f:
                    f.write(cover_image_upload.getbuffer())
                st.success(f"Cover-Bild '{cover_image_upload.name}' erfolgreich hochgeladen und gespeichert.")

            # Schreiben der Standard Attribute
            new_version = Version(name=name, instructions=instructions, complexity=complexity, cover_image_path=f'{image_directory}/cover_{cover_image_upload.name}', time_limit=time_limit)
            session.add(new_version)
            session.commit()

            # Bildpfade der hochgeladenen Bilder im Versions-Objekt hinterlegen
            for uploaded_file in uploaded_files:
                # Dateipfad für das Bild definieren
                image_path = os.path.join(image_directory, uploaded_file.name)

                # Speichern der Datei
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"Bild '{uploaded_file.name}' erfolgreich hochgeladen und gespeichert.")
                # Für jedes Bild einen neuen Datensatz erstellen
                new_image = Image(version_id=new_version.id, image_path=image_path)
                session.add(new_image)

            session.commit()

            st.success("Neue Version hinzugefügt!")

            # Warte 2 Sekunden und Lade dann die Seite neu
            time.sleep(2)

            # Seite neu laden, damit die neue Version auch sofort in der Erstellung neuer Aufgabenprofile genutzt werden kann
            st.rerun()

        else:
            st.warning("Alle Angaben befüllen!")


# Version löschen
def version_delete():

    st.subheader("Version löschen")

    # Liste der vorhandenen Versionen aus der Datenbank abrufen
    versions = session.query(Version).all()
    version_names = [version.name for version in versions]

    # Dropdown-Menü für die Auswahl der zu löschenden Version anzeigen
    version_to_delete = st.selectbox("Wähle eine Version zum Löschen aus", version_names)

    # Durch das Löschen der Version werden auch die verknüpften Images gelöscht, bevor dies geschieht, soll der Ordner gelöscht werden
    @event.listens_for(session, 'before_flush')
    def before_flush(session, flush_context, instances):
        for instance in session.deleted:
            if isinstance(instance, Version):
                delete_image_folder(instance.images)

    # Wenn die Version gelöscht werden soll
    if st.button("Version löschen"):

        # Query zum Löschen der Version aus der Datenbank
        version_to_delete_obj = session.query(Version).filter_by(name=version_to_delete).first()

        # Lösche das gefundene Objekt
        if version_to_delete_obj:
            session.delete(version_to_delete_obj)
            session.commit()
            st.success(f"Version '{version_to_delete}' erfolgreich gelöscht.")
            time.sleep(2)
            st.rerun()
        else:
            st.warning(f"Version '{version_to_delete}' nicht gefunden.")


def task_profile_edit():

    st.subheader("Aufgabenprofil bearbeiten")

    # Lade Aufgabenprofile und Versions
    task_profiles = get_task_profiles()
    versions = get_versions()
    task_profile_ids = [tp.id for tp in task_profiles]
    task_profile_dict = {tp.id: tp for tp in task_profiles}

    # Auswahl für Aufgabenprofile
    selected_task_profile_id = st.selectbox("Wähle ein Aufgabenprofil aus", task_profile_ids, format_func=lambda x: task_profile_dict[x].name)

    if selected_task_profile_id:
        selected_task_profile = task_profile_dict[selected_task_profile_id]

        # Bearbeitung der Aufgabenprofile
        name = st.text_input("Name", value=selected_task_profile.name)

        # Auswahl der Versions und Eingabe der Mengen
        selected_versions_counts = []
        for version in versions:
            is_selected = any(version.id == v.id for v in selected_task_profile.versions)  # Prüfe, ob die jeweilige Version schon im Aufgabenprofil genutzt
            count = next((rc.count for rc in selected_task_profile.required_counts if rc.version_id == version.id), 1)  # Gib ersten Wert zurück, der Bedingung erfüllt, sonst 1
            if st.checkbox(f"{version.name}", value=is_selected, key=f"check_{version.id}_edit"):
                count = st.number_input(f"Menge für {version.name}", min_value=1, value=count, key=f"count_{version.id}_edit")
                selected_versions_counts.append((version.id, count))

        if st.button("Änderungen speichern"):
            # Update Task Profile
            update_task_profile(selected_task_profile_id, name, selected_versions_counts)
            st.success("Aufgabenprofil wurde erfolgreich aktualisiert.")


def task_profile_creation():
    # Neue Aufgabenprofile hinzufügen
    st.subheader("Neues Aufgabenprofil hinzufügen")
    profile_name = st.text_input("Name des Aufgabenprofils")

    # Ziehe zunächst alle vorhandenen Versionen
    versions = session.query(Version).all()
    selected_versions_counts = []

    assign_to_all = st.checkbox("Für alle Nutzer freischalten")

    st.write("")
    st.write("Versionen")

    if versions:
        for version in versions:
            if st.checkbox(f"{version.name}", key=f"check_{version.id}_creation"):
                count = st.number_input(f"Menge für {version.name}", min_value=1, value=1, key=f"count_{version.id}_creation")
                selected_versions_counts.append((version.id, count))

    # Falls neues Aufgabenprofil erstellt wird
    if st.button("Neues Aufgabenprofil erstellen"):

        # Nur, wenn alle nötigen Angaben getätigt wurden
        if profile_name and selected_versions_counts:

            new_task_profile = TaskProfile(name=profile_name)
            session.add(new_task_profile)
            session.commit()

            for version_id, count in selected_versions_counts:
                version = session.query(Version).filter_by(id=version_id).first()
                if version:
                    new_task_profile.versions.append(version)
                    required_count = TaskProfileRequiredCount(task_profile_id=new_task_profile.id, version_id=version_id, count=count)
                    session.add(required_count)

            session.commit()
            st.success(f"Neues Aufgabenprofil '{profile_name}' erstellt!")

            # Hinzufügen des Profiles, wenn Checkbox
            if assign_to_all:
                users = session.query(User).all()
                user_ids = [user.id for user in users]
                success, message = add_task_profile_to_users(user_ids, profile_name)
                if success:
                    st.success(message)
                else:
                    st.error(message)

            time.sleep(2)
            st.rerun()
        else:
            st.error(
                "Bitte geben Sie einen Namen für das neue Aufgabenprofil ein und wählen Sie mindestens eine Version aus.")


def task_profile_delete():
    # Aufgabenprofil löschen
    st.subheader("Aufgabenprofil löschen")

    # Liste der vorhandenen Aufgabenprofile aus der Datenbank abrufen
    profiles = session.query(TaskProfile).all()
    profile_names = [profile.name for profile in profiles]

    # Dropdown-Menü für die Auswahl des zu löschenden Aufgabenprofils anzeigen
    profile_to_delete = st.selectbox("Wähle ein Aufgabenprofil zum Löschen aus", profile_names)

    if st.button("Aufgabenprofil löschen"):
        # Query zum Löschen des Aufgabenprofils aus der Datenbank
        profile_to_delete_obj = session.query(TaskProfile).filter_by(name=profile_to_delete).first()

        if profile_to_delete_obj:
            session.delete(profile_to_delete_obj)
            session.commit()
            st.success(f"Aufgabenprofil '{profile_to_delete}' erfolgreich gelöscht.")
            time.sleep(2)
            st.rerun()
        else:
            st.warning(f"Aufgabenprofil '{profile_to_delete}' nicht gefunden.")


@st.dialog("Aufgaben zurücksetzen")
def delete_tasks_dia():
    st.write(
        "Sollen die gefilterten Tasks gelöscht werden? Dieser Vorgang kann nicht rückgängig gemacht werden.")
    if st.button("Ja, Löschen!"):
        success, error = delete_filtered_tasks(st.session_state.filtered_task_ids)
        if success:
            st.success("Gefilterte Tasks wurden erfolgreich gelöscht.")
        else:
            st.error(f"Fehler beim Löschen der gefilterten Tasks: {error}")
        st.session_state.show_confirm_delete_tasks = False
        st.rerun()


def delete_tasks():

    # Aufgaben löschen
    st.subheader("Aufgaben löschen")

    task_data = load_task_data()

    filtered_task_data = filter_dataframe(task_data)

    event = st.dataframe(
        filtered_task_data,
        width='stretch',
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )

    st.write("Zur Löschung ausgewählte Aufgaben")

    tasks_to_delete = event.selection.rows
    filtered_df = filtered_task_data.iloc[tasks_to_delete]
    st.dataframe(
        filtered_df,
        width='stretch',
    )

    # Button zum Löschen gefilterter Tasks
    if st.button("Gefilterte Tasks löschen"):
        if not filtered_df.empty:

            #confirm_delete_tasks
            st.session_state.filtered_task_ids = filtered_df['task_id'].tolist()
            delete_tasks_dia()
        else:
            st.warning("Keine Tasks entsprechen den Filterkriterien.")


def version_status():
    # Versionen laden
    versions = session.query(Version).all()

    st.header("Versionen deaktivieren/aktivieren")

    # Darstellen der Versionen und des jeweiligen Status über Checkbox
    for version in versions:
        is_active = st.checkbox(f"{version.name} aktivieren", value=version.active)
        if is_active != version.active:
            # Aktualisieren des Status für eine Version, wenn sich ihr Wert ändert
            toggle_version_status(version.id, is_active)


def task_profile_status():
    # Profile laden
    task_profiles = session.query(TaskProfile).all()

    st.header("Aufgabenprofil deaktivieren/aktivieren")
    for task_profile in task_profiles:
        is_active = st.checkbox(f"{task_profile.name} aktivieren", value=task_profile.active)
        if is_active != task_profile.active:
            toggle_task_profile_status(task_profile.id, is_active)

# Allgemeine Einstellungen für den Arbeitsplatz eines Nutzers

def general_settings():

    st.subheader("Allgemeine Einstellungen")

    random_profile = st.toggle("Im Arbeitsplatz des Nutzers soll ein zufälliges Aufgabenprofil ausgewählt werden: ", value=eval(get_setting('random_profile', "False")))

    save_setting('random_profile', str(random_profile))

    random_mode = st.session_state.random_mode = st.toggle("Im Arbeitsplatz des Nutzers soll ein zufälliger Spielmodus ausgewählt werden: ", value=eval(get_setting('random_mode', "False")))

    save_setting('random_mode', str(random_mode))

    #print(f"randmom_profile: {random_profile}, {get_setting('random_profile')}")
    #print(f"randmom_mode: {random_mode}, {get_setting('random_mode')}")


def component_creation():

    st.subheader("Neues Bauteil hinzufügen")

    st.write("Um ein neues Bauteil einzupflegen, fügen Sie zunächst eine Abbildung dieses Teils an")

    component_image = st.file_uploader("Bauteil-Bild", accept_multiple_files=False)

    if component_image:
        bytes_data = component_image.read()
        st.image(bytes_data)

        component_name = st.text_input(
            "Name des Bauteils 👇",
            placeholder='Benenne die Komponente mit einem einzigartigen Namen',
        )

        if st.button("Bauteil hinzufügen"):

            # Prüfe, ob Name einzigartig
            if not is_component_unique(component_name):
                st.toast("**Der Versionsname ist bereits vorhanden. Bitte wählen Sie einen anderen Namen**", icon="⚠️")
            else:

                # Dateipfad für das Bild definieren
                image_directory = f'images/components'
                image_path = os.path.join(image_directory, component_image.name)

                # Speichern der Datei
                with open(image_path, "wb") as f:
                    f.write(component_image.getbuffer())
                create_new_component(component_name, image_path)
                st.success(f"Bild '{component_image.name}' erfolgreich hochgeladen und in Bauteil {component_name} erstellt.")

                # Warte 2 Sekunden und Lade dann die Seite neu
                time.sleep(2)

                st.session_state.clear_inputs = True

                # Seite neu laden, damit die neue Componente auch sofort in der Erstellung neuer Bauteillisten genutzt werden kann
                st.rerun()


def component_list_creation():
    st.subheader("Neue Bauteilliste erstellen")

    component_list_name = st.text_input("Name der Bauteilliste")

    # Ziehe zunächst alle vorhandenen Bauteile
    components = session.query(Component).all()
    selected_components_counts = []

    st.write("")
    st.write("Bauteile")

    if components:
        for component in components:
            if st.checkbox(f"{component.name}", key=f"check_{component.id}_component_list_creation"):
                if component.component_image_path:
                    st.image(component.component_image_path)
                count = st.number_input(f"Menge für {component.name}", min_value=1, value=1, key=f"count_{component.id}_component_list_creation")
                selected_components_counts.append((component.id, count))

    # Falls neues Aufgabenprofil erstellt wird
    if st.button("Neue Bauteilliste erstellen"):

        # Nur, wenn alle nötigen Angaben getätigt wurden
        if component_list_name and selected_components_counts:
            
            # Neue Bauteilliste erstellen
            new_component_list = ComponentList(name=component_list_name)
            session.add(new_component_list)
            session.commit()

            # Füge benötigte Counts hinzu
            for component_id, count in selected_components_counts:
                component = session.query(Component).filter_by(id=component_id).first()
                if component:
                    # Erstelle die Zuordnung in der ComponentListRequiredCount-Tabelle
                    required_count = ComponentListRequiredCount(component_list_id=new_component_list.id, component_id=component_id, count=count)
                    session.add(required_count)

            session.commit()  # Speichere die Counts

            st.success(f"Neue Bauteilliste '{component_list_name}' erstellt!")

            time.sleep(2)
            st.rerun()
        else:
            st.error(
                "Bitte geben Sie einen Namen für die Bauteilliste an und wählen Sie mindestens ein Bauteil aus")



def component_list_assign():

    st.subheader("Bauteilliste zuweisen")

    # Alle freien Bauteillisten
    free_component_lists = session.query(ComponentList).filter(ComponentList.version_id == None).all()

    # Alle Versionen
    versions = session.query(Version).all()

    # Alle zugewiesenen Bauteillisten
    assigned_component_lists = session.query(ComponentList).filter(ComponentList.version_id != None).all()

    # Mapping von version_id zu ihren zugewiesenen ComponentLists
    assigned_component_lists_by_version = {cl.version_id: cl for cl in assigned_component_lists}

    st.session_state.update_lists = False

    if versions:

        for version in versions:

            cols = st.columns([1,1,1])

            assigned_to_version = assigned_component_lists_by_version.get(version.id)
            available_lists = free_component_lists.copy()

            if assigned_to_version:
                available_lists.append(assigned_to_version)  # Die zugewiesene Liste als Option hinzufügen
                available_lists.append({"name": "Leeren", "id": None})

            # Setze die zugewiesene Liste als Standard, falls vorhanden
            default_value = assigned_to_version if assigned_to_version else None

            cols[0].write(version.name)
            cols[1].write("zugewiesene Bauteilliste: ")

            session_key = f'assign_component_list_to_{version.name}'

            # Dropdown für die Zuordnung von ComponentLists zu dieser Version
            selected_component_list = cols[2].selectbox(
                "Bauteilliste",
                available_lists,  # Liste der auswählbaren Bauteillisten
                format_func=lambda cl: cl.name if hasattr(cl, 'name') else cl.get('name', cl),
                index=available_lists.index(default_value) if default_value else None,
                key=f'{session_key}_select',
                placeholder="Wähle eine Liste",
                #on_change=update_lists(free_component_lists)
            )

            if selected_component_list:

                # Prüfe, ob "Leeren" ausgewählt wurde
                if isinstance(selected_component_list, dict):
                    
                    # Änderungen in der Datenbank speichern
                    if assigned_to_version:
                        assigned_to_version.version_id = None  # Alte Zuweisung entfernen
                        session.commit()

                else:
                    database_selected_component_list = session.query(ComponentList).filter(ComponentList.id == selected_component_list.id).one_or_none()

                    if database_selected_component_list:

                        database_selected_component_list.version_id = version.id  # Neue Zuweisung

                        session.commit()

                free_component_lists = session.query(ComponentList).filter(ComponentList.version_id == None).all()


def component_list_delete():
    st.subheader("Bauteilliste löschen")

    # Alle Bauteillisten abrufen
    component_lists = session.query(ComponentList).all()

    if component_lists:
        # Auswahlbox für die Bauteilliste
        component_list_names = {cl.id: cl.name for cl in component_lists}
        selected_list_id = st.selectbox("Wähle eine Bauteilliste zum Löschen:",
                                        options=list(component_list_names.keys()),
                                        format_func=lambda x: component_list_names[x])

        if st.button("Bauteilliste löschen"):
            # Überprüfen, ob eine Liste ausgewählt wurde
            if selected_list_id:
                # Lösche alle zugehörigen Einträge in der ComponentListRequiredCount-Tabelle
                session.query(ComponentListRequiredCount).filter_by(component_list_id=selected_list_id).delete()

                # Lösche die Bauteilliste selbst
                component_list_to_delete = session.query(ComponentList).filter_by(id=selected_list_id).first()
                if component_list_to_delete:
                    session.delete(component_list_to_delete)
                    session.commit()  # Änderungen speichern

                    st.success(f"Bauteilliste '{component_list_names[selected_list_id]}' erfolgreich gelöscht!")

                    time.sleep(2)

                    st.rerun()

                else:
                    st.error("Bauteilliste nicht gefunden.")
            else:
                st.error("Bitte wähle eine Bauteilliste aus.")
    else:
        st.write("Keine Bauteillisten vorhanden.")


def component_edit():

    st.subheader("Bauteil bearbeiten/löschen")

    # Alle Bauteile abrufen
    components = session.query(Component).all()

    if components:
        # Auswahlbox für die Bauteile
        component_names = {cl.id: cl.name for cl in components}
        selected_component_id = st.selectbox("Wähle eine Bauteil zum bearbeiten/löschen",
                                        options=list(component_names.keys()),
                                        format_func=lambda x: component_names[x])


        if selected_component_id:

            # Bauteil aus der Datenbank
            component = session.query(Component).filter_by(id=selected_component_id).first()

            edit_delete_radio = st.radio(
                "Was möchte du mit dem Bauteil machen",
                ["Bearbeiten", "Löschen"],
                index=0,
                horizontal=True
            )

            if edit_delete_radio == "Bearbeiten":

                # Zeige das Bild des Bauteils, falls vorhanden
                if component.component_image_path:
                    st.image(component.component_image_path)

                # Zeige den aktuellen Namen und ermögliche das Bearbeiten
                new_component_name_input = st.text_input(
                    "Neuer Bauteil-Name 👇",
                    value=component.name
                )

                # Überprüfen und Speichern des neuen Namens
                if st.button("Speichern"):
                    if new_component_name_input and new_component_name_input != component.name:
                        if not is_component_unique(new_component_name_input):
                            st.warning("Der Name ist bereits vergeben. Bitte wählen Sie einen anderen.")
                        else:
                            # Aktualisieren der Komponente in der Datenbank
                            update_component(selected_component_id, new_component_name_input)
                            st.success(f"Bauteil {new_component_name_input} erfolgreich aktualisiert.")
                            time.sleep(2)
                            st.rerun()
                    elif new_component_name_input == component.name:
                        st.info("Der Name ist unverändert.")



            if edit_delete_radio == "Löschen":

                # Zeige das Bild des Bauteils, falls vorhanden
                if component.component_image_path:
                    st.image(component.component_image_path)

                if st.button("Löschen des Bauteils"):
                    delete_component(selected_component_id)
                    st.success(f"Bautteil {component_names[selected_component_id]} gelöscht.")

                    time.sleep(2)
                    st.rerun()

    else:
        st.write("Keine Bauteillisten vorhanden.")


def component_list_edit():
    st.subheader("Bauteilliste bearbeiten")

    # Alle Bauteillisten abrufen
    component_lists = session.query(ComponentList).all()

    if component_lists:
        # Auswahlbox für die Bauteilliste
        component_list_names = {cl.id: cl.name for cl in component_lists}
        selected_list_id = st.selectbox("Wähle eine Bauteilliste zum Bearbeiten",
                                        options=list(component_list_names.keys()),
                                        format_func=lambda x: component_list_names[x],
                                        key="selected_list_id_edit")
        if selected_list_id:

            # Hole die Bauteilliste und die bereits verknüpften Bauteile
            component_list = session.query(ComponentList).filter_by(id=selected_list_id).first()
            if not component_list:
                st.error("Bauteilliste nicht gefunden.")
                return

            # Vorhandenen Namen anzeigen und Möglichkeit zur Bearbeitung
            component_list_name = st.text_input("Name der Bauteilliste", value=component_list.name)

            # Alle Bauteile holen
            components = session.query(Component).all()
            selected_components_counts = []

            st.write("")
            st.write("Bauteile")

            # Erstelle eine Liste von bereits enthaltenen Komponenten und deren Mengen
            existing_components_counts = {req_count.component_id: req_count.count for req_count in
                                          component_list.required_counts}

            if components:
                for component in components:
                    # Überprüfen, ob die Komponente bereits in der Bauteilliste enthalten ist
                    is_checked = component.id in existing_components_counts

                    # Erstelle Checkbox für das Bauteil, falls es bereits in der Liste ist, setze den Wert auf True
                    if st.checkbox(f"{component.name}", value=is_checked, key=f"check_{component.id}_component_list_edit"):
                        # Zeige das Bild der Komponente, falls vorhanden
                        if component.component_image_path:
                            st.image(component.component_image_path)

                        # Setze die aktuelle Menge als Standardwert, falls die Komponente bereits in der Liste ist
                        count = st.number_input(
                            f"Menge für {component.name}",
                            min_value=1,
                            value=existing_components_counts.get(component.id, 1),
                            key=f"count_{component.id}_component_list_edit"
                        )
                        selected_components_counts.append((component.id, count))

            # Änderungen speichern
            if st.button("Änderungen speichern", key="bauteillisten_änderungen_speichern"):
                if component_list_name and selected_components_counts:
                    # Aktualisiere den Namen der Bauteilliste
                    component_list.name = component_list_name

                    # Setze die vorhandenen Bauteile und Mengen neu
                    existing_component_ids = set(existing_components_counts.keys())
                    new_component_ids = {comp_id for comp_id, _ in selected_components_counts}

                    # Bauteile, die abgewählt wurden, aus der Bauteilliste entfernen
                    for comp_id in existing_component_ids - new_component_ids:
                        session.query(ComponentListRequiredCount).filter_by(
                            component_list_id=component_list.id,
                            component_id=comp_id
                        ).delete()

                    # Vorhandene Bauteile aktualisieren oder neue hinzufügen
                    for comp_id, count in selected_components_counts:
                        existing_entry = session.query(ComponentListRequiredCount).filter_by(
                            component_list_id=component_list.id,
                            component_id=comp_id
                        ).first()
                        if existing_entry:
                            existing_entry.count = count  # Menge aktualisieren
                        else:
                            # Neue Zuordnung in ComponentListRequiredCount erstellen
                            new_entry = ComponentListRequiredCount(
                                component_list_id=component_list.id,
                                component_id=comp_id,
                                count=count
                            )
                            session.add(new_entry)

                    session.commit()  # Speichere alle Änderungen in der Datenbank

                    st.success(f"Bauteilliste '{component_list_name}' erfolgreich aktualisiert!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Bitte geben Sie einen Namen für die Bauteilliste an und wählen Sie mindestens ein Bauteil aus.")


def version_step_componentes():
    st.title("Bauteilverwaltung für Schritte")

    # Version auswählen
    versions = session.query(Version).all()
    version_names = {version.id: version.name for version in versions}
    selected_version_id = st.selectbox(
        "Wähle eine Version",
        options=list(version_names.keys()),
        format_func=lambda x: version_names[x],
        key="selected_version_id"
    )

    # Gewählte Version abrufen
    selected_version = session.query(Version).filter_by(id=selected_version_id).first()

    # Schritt auswählen
    if selected_version:

        check_component_discrepancies(selected_version.id)

        if selected_version.cover_image_path:
            st.image(selected_version.cover_image_path)

        steps = selected_version.images  # Images der Version abrufen
        step_names = {step.id: f"{step.id} - {step.image_anleitung}" for step in steps}
        selected_step_id = st.selectbox(
            "Wähle einen Schritt",
            options=list(step_names.keys()),
            format_func=lambda x: step_names[x],
            key="selected_step_id"
        )

        # Wenn die Version oder der Schritt gewechselt wird, Bauteil-Einträge aus dem Session-State entfernen
        if (
                "previous_version_id" in st.session_state and st.session_state.previous_version_id != selected_version_id
        ) or (
                "previous_step_id" in st.session_state and st.session_state.previous_step_id != selected_step_id
        ):
            # Checkboxen und Anzahl-Eingaben aus dem Session-State löschen
            for key in list(st.session_state.keys()):
                if key.startswith("check_step_componentes_") or key.startswith("count_step_componentes_"):
                    del st.session_state[key]

        # Aktualisiere die previous_version_id und previous_step_id
        st.session_state.previous_version_id = selected_version_id
        st.session_state.previous_step_id = selected_step_id

        # Gewählten Schritt abrufen
        selected_step = session.query(Image).filter_by(id=selected_step_id).first()

        # Bauteile und Anzahl auswählen
        if selected_step:

            #if selected_step.image_path:
            #    st.image(selected_step.image_path)

            components = session.query(Component).all()

            # Vorhandene Bauteile für diesen Schritt abrufen
            existing_requirements = {req.component_id: req.count for req in selected_step.required_components}

            selected_components_counts = []

            st.write("")
            st.subheader("Bauteile für den Schritt")

            # Auswahl der Bauteile mit Checkbox und NUmber-Input
            for component in components:
                
                # Überprüfen, ob das Bauteil bereits für den Schritt angegeben ist
                is_checked = component.id in existing_requirements

                # Checkbox für das Bauteil, mit Initialisierung des Session-States
                if f"check_step_componentes_{component.id}_step" not in st.session_state:
                    st.session_state[f"check_step_componentes_{component.id}_step"] = is_checked

                if f"count_step_componentes_{component.id}_step" not in st.session_state:
                    st.session_state[f"count_step_componentes_{component.id}_step"] = existing_requirements.get(component.id, 1)

                # Checkbox und NUmber-Input anzeigen
                if st.checkbox(f"{component.name}", key=f"check_step_componentes_{component.id}_step"):

                    # Bild des Bauteils anzeigen, falls vorhanden
                    if component.component_image_path:
                        st.image(component.component_image_path, caption=component.name)

                    # NUmber-Input für das Bauteil
                    count = st.number_input(
                        f"Menge für {component.name}",
                        min_value=1,
                        value=st.session_state[f"count_step_componentes_{component.id}_step"],
                        key=f"count_step_componentes_{component.id}_step"
                    )
                    
                    selected_components_counts.append((component.id, count))

            # Änderungen speichern
            if st.button("Änderungen speichern", key="version_step_componentes_save"):

                # Alle bisherigen Bauteile für diesen Schritt löschen
                session.query(TaskComponentRequirement).filter_by(image_id=selected_step.id).delete()

                # Neue Bauteilliste für den Schritt speichern
                for component_id, count in selected_components_counts:
                    requirement = TaskComponentRequirement(
                        image_id=selected_step.id,
                        component_id=component_id,
                        count=count
                    )
                    session.add(requirement)

                session.commit()
                st.success("Änderungen wurden erfolgreich gespeichert!")

                time.sleep(2)
                st.rerun()



# -------- Design der Seite (Aufbau und Aussehen)  --------

st.title("Aufgaben Management")

general_settings()

st.divider()

# Tabs für Versionen
version_tab1, version_tab2, version_tab2_5, version_tab3, version_tab4 = st.tabs(["Neue Version erstellen", "Version bearbeiten", "Version - Bauteile zuweisen","Version deaktivieren/aktivieren (?)", "Version löschen (nicht empfohlen)"])


st.divider()

# Tabs für Aufgabenprofile
profile_tab1, profile_tab2, profile_tab3, profile_tab4 = st.tabs(["Neues Aufgabenprofil erstellen", "Aufgabenprofil bearbeiten", "Aufgabenprofil deaktivieren/aktivieren", "Aufgabenprofil löschen"])


with version_tab1:
    version_creation_2()
with version_tab2:
    with st.container(border=True):
        version_edit_2()

with version_tab2_5:
    with st.container(border=True):
        version_step_componentes()

with version_tab3:
    with st.container(border=True):
        version_status()
with version_tab4:
    with st.container(border=True):
        version_delete()

st.divider()

with profile_tab1:
    with st.container(border=True):
        task_profile_creation()
with profile_tab2:
    with st.container(border=True):
        task_profile_edit()
with profile_tab3:
    with st.container(border=True):
        task_profile_status()
with profile_tab4:
    with st.container(border=True):
        task_profile_delete()


component_tab1, component_tab1_5, component_tab2, component_tab2_5, component_tab3, component_tab4 = st.tabs(["Neues Bauteil hinzufügen", "Bauteil bearbeiten/löschen", "Neue Bauteilliste erstellen", "Bauteilliste bearbeiten", "Bauteilliste zuteilen", "Bauteilliste löschen"])

with component_tab1:
    with st.container(border=True):
        component_creation()

with component_tab1_5:
    with st.container(border=True):
        component_edit()

with component_tab2:
    with st.container(border=True):
        component_list_creation()

with component_tab2_5:
    with st.container(border=True):
        component_list_edit()

with component_tab3:
    with st.container(border=True):
        component_list_assign()

with component_tab4:
    with st.container(border=True):
        component_list_delete()

st.divider()

with st.container(border=True):
    delete_tasks()
