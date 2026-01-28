import shutil
import pandas as pd
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from data.database_code import User, Image, Task, TaskStep, TaskProfile, Version, TaskProfileRequiredCount, Setting, \
    VersionHistory, ImageHistory, Component, ComponentListRequiredCount, component_list_components_association, ComponentList, ComponentListRequiredCount, TaskComponentRequirement
from data.database_code import session
from datetime import datetime
import os
import streamlit as st
import json

def add_user(session, firstname, lastname, nickname, age, skill, selected_task_profiles):

    """
    Hinzufügen eines neuen Nutzers

    Args:
       session: Aktuelle Datenbank-Session,
       firstname: Gewünschter Vorname,
       lastname: Gewünschter Nachname,
       nickname: Nickname des Nutzer -> Dieser muss einzigartig sein!,
       age: Alter des Nutzers,
       skill: Skill des Nutzers,
       selected_task_profiles: Die dem Nutzer zugeordneten Profile

    Raises:
       ValueError, falls Nickname nicht einzigartig ist -> Keine Erstellung des Nutzers
    """

    # Überprüfe, ob der Nickname bereits existiert
    existing_user = session.query(User).filter_by(nickname=nickname).first()
    if existing_user:
        raise ValueError(f"Der Nickname '{nickname}' ist bereits vergeben. Bitte wähle einen anderen.")

    # Falls der Nickname einzigartig ist, erstelle einen neuen Benutzer
    new_user = User(firstname=firstname, lastname=lastname, nickname=nickname, age=age, skill=skill, task_profiles=[session.query(TaskProfile).filter_by(id=tp_id).first() for tp_id in selected_task_profiles])
    session.add(new_user)
    session.commit()


# Funktion zum Laden der Bilder für eine gegebene Version
def load_images(version_id):
    """
    Funktion zum Laden der Bilder für eine gegebene Version

    Args:
       version_id: ID der Version, von welcehr die Vilder geladen werden soll

    Returns:
        Die Für die Version gefunden Bilder
    """

    # letzte Revision der Version abrufen
    version_history = session.query(VersionHistory).filter_by(version_id=version_id).order_by(VersionHistory.revision.desc()).first()

    if version_history:
        # Bilder der letzten Revision abrufen
        images = session.query(ImageHistory).filter_by(version_history_id=version_history.id).all()
        if images:
            return images
        else:
            st.warning(f"Keine Bilder für Version {version_id} in Revision {version_history.revision} gefunden.")
            return []
    else:
        st.warning(f"Keine Revisionen für Version {version_id} gefunden.")
        return []

    return images


# Funktion zum Speichern der Aufgabe in der Datenbank
def save_task(user_id, version_id, time, start_timestamp, game_mode):
    """
    Funktion zum Speichern der Aufgabe in der Datenbank

    Args:
        user_id: Die ID des Nutzers,
        version_id: Die ID der Version,
        time: Zeit,
        start_timestamp: Start Timestamp der Aufgabe,
        game_mode: Der Spielmodus in dem der Nutzer die Aufgabe durchführt(ohne timer -> classic, mit timer -> timer, mit Countdown ->countdown)

    Returns:
        Falls die Erstellung der neuen Aufgabe erfolgreich war, wird das Aufgaben-Objekt zurückgegeben
    """

    # aktuelle Revision der Version
    latest_version_history = session.query(VersionHistory).filter_by(version_id=version_id).order_by(VersionHistory.revision.desc()).first()

    if not latest_version_history:
        st.error(f"Keine Revisionen für Version {version_id} gefunden.")
        return

    # Neuen Task-Eintrag erstellen
    new_task = Task(
        user_id=user_id,
        version_id=version_id,
        version_history_id=latest_version_history.id,  # Verknüpfung zur VersionHistory
        time=time,
        game_mode=game_mode,
        start_timestamp=start_timestamp,
        end_timestamp=datetime.now()
    )

    session.add(new_task)
    session.commit()

    return new_task


# Funktion zum Speichern der Montageschritte in der Datenbank
def save_task_steps(task_id, steps_times):

    """
    Funktion zum Speichern der Montageschritte in der Datenbank

    Args:
        task_id: Die ID der Aufgabe,
        steps_times: Die benötigten Zeiten für die TaskSteps
    """

    for idx, step in enumerate(steps_times):
        new_step = TaskStep(
            task_id=task_id,
            step_number=idx + 1,
            start_time=step['start_time'],
            end_time=step['end_time'],
            time_spent=step['time_spent']
        )
        session.add(new_step)
    session.commit()

# Erstelle neue Version
def create_new_version(name, description, complexity, cover_image_path, time_limit, images):
    """
    Funktion zum Erstellen einer neuen Version

    Args:
        name: Name der Version,
        description: Kurzbeschreibung der Version,
        complexity: Komplexität der Version,
        cover_image_path: Pfad zum Cover-Bild,
        time_limit: Zeitlimit für die Version
        images: Pfade und Beschreibungen zu den Bildern für die Version

    Returns:
        Falls die Erstellung der neuen Version erfolgreich war, wird das Versions-Objekt zurückgegeben
    """

    # Neue Version anlegen
    new_version = Version(
        name=name,
        description=description,
        complexity=complexity,
        cover_image_path=cover_image_path,
        time_limit=time_limit
    )

    session.add(new_version)
    session.commit()

    # Erste Revision für die Version in der Historie speichern
    new_version_history = VersionHistory(
        version_id=new_version.id,
        revision=1,
        name=name,
        description=description,
        complexity=complexity,
        time_limit=time_limit
    )
    session.add(new_version_history)
    session.commit()

    # Bilder für die neue Version speichern
    for image in images:
        new_image = Image(
            version_id=new_version.id,
            image_path=image['image_path'],
            image_anleitung=image['image_anleitung']
        )
        session.add(new_image)

        # Neue ImageHistorie für jedes Bild
        new_image_history = ImageHistory(
            image_id=new_image.id,
            version_history_id=new_version_history.id,  # Verknüpfung zur VersionHistory
            revision=1,
            image_path=image['image_path'],
            image_anleitung=image['image_anleitung']
        )
        session.add(new_image_history)

    session.commit()
    return new_version


#Aktualisiere Version
def update_version(version_id, new_name, new_description, new_complexity, new_cover_image_path, new_time_limit, updated_images):
    """
    Funktion zum Aktualisieren einer Version

    Args:
        version_id: ID der zu aktualisierenden Version
        new_name: neuer Name der Version,
        new_description: neue Kurzbeschreibung der Version,
        new_complexity: neue Komplexität der Version,
        new_cover_image_path: neuer Pfad zum Cover-Bild,
        new_time_limit: neues Zeitlimit für die Version
        updated_images: aktualisierte Pfade und Beschreibungen zu den Bildern für die Version -> Wenn leer, dann werden die alten Bilder übernommen

    """

    # aktuelle Version abrufen
    version = session.query(Version).filter_by(id=version_id).first()

    if not version:
        st.error(f"Version mit ID {version_id} nicht gefunden.")
        return

    # aktuelle Revision abrufen und neue Revisionsnummer festlegen
    latest_revision = session.query(VersionHistory).filter_by(version_id=version_id).order_by(VersionHistory.revision.desc()).first()
    new_revision_number = (latest_revision.revision + 1) if latest_revision else 1

    # Neues Verzeichnis für die neue Revision erstellen
    version_directory = f'images/{new_name}/revision_{new_revision_number}'
    if not os.path.exists(version_directory):
        os.makedirs(version_directory)

    # Versionseigenschaften aktualisieren
    version.name = new_name

    if new_description:
        version.description = new_description
    version.complexity = new_complexity
    version.time_limit = new_time_limit

    # Cover-Bild aktualisieren
    if new_cover_image_path:
        version.cover_image_path = new_cover_image_path


    # Neue Revision für die Version speichern
    new_version_history = VersionHistory(
        version_id=version_id,
        revision=new_revision_number,
        name=new_name,
        description=new_description if new_description else version.description,
        complexity=new_complexity,
        time_limit=new_time_limit
    )
    session.add(new_version_history)
    session.commit()

    if updated_images != []:

        # Bilder aktualisieren und in den neuen Revisionsordner speichern
        for image in updated_images:
            image_file_name = os.path.basename(image['image_path'])  # Extrahiere den Bildnamen
            new_image_path = os.path.join(version_directory, image_file_name)

            # Bild speichern
            shutil.copy(image['image_path'], new_image_path)
            os.remove(image['image_path'])
            st.success(f"'{image_file_name}' erfolgreich hochgeladen und gespeichert.")

            # Neue Image-Objekte für die aktuelle Version erstellen
            new_image = Image(
                version_id=version_id,
                image_path=new_image_path,
                image_anleitung=image['image_anleitung']
            )
            session.add(new_image)
            session.commit()

            # Historie für das Bild erstellen
            new_image_history = ImageHistory(
                image_id=new_image.id,
                version_history_id=new_version_history.id,
                revision=new_revision_number,
                image_path=new_image_path,
                image_anleitung=image['image_anleitung']
            )
            session.add(new_image_history)

    # Falls keine neuen Bilder hochgeladen wurden, dann Bilder der letzten Revision kopieren
    else:

        previous_images = session.query(ImageHistory).filter_by(version_history_id=latest_revision.id).all()

        if previous_images:
            for prev_image in previous_images:

                # Pfad zum alten Bild und neuer Pfad der neuen Revision
                old_image_path = prev_image.image_path
                image_file_name = os.path.basename(old_image_path)
                new_image_path = os.path.join(version_directory, image_file_name)

                # Kopieren der Bilddateien
                shutil.copy(old_image_path, new_image_path)

                # Ein neues Image für die aktuelle Version erstellen
                new_image = Image(
                    version_id=version_id,
                    image_path=new_image_path,
                    image_anleitung=prev_image.image_anleitung  # Anleitungstext beibehalten
                )
                session.add(new_image)
                session.commit()

                # Neue Historie für das kopierte Bild erstellen
                new_image_history = ImageHistory(
                    image_id=new_image.id,
                    version_history_id=new_version_history.id,
                    revision=new_revision_number,
                    image_path=new_image_path,
                    image_anleitung=prev_image.image_anleitung
                )
                session.add(new_image_history)


    session.commit()

    st.success(f"Version {version_id} erfolgreich auf Revision {new_revision_number} aktualisiert.")


def load_task_data(user_id=None):
    """
    Lade alle Task-Daten aus der lokalen Datenbank

    Args:
       user_id: optionaler Parameter zum filtern von spezifischen Nutzer-Aufgaben (sonst alle Daten)

    Returns:
       pd.DataFrame: Alle gefunden Daten
    """

    if user_id is not None:
        tasks = session.query(Task).filter(Task.user_id == user_id).all()
    else:
        tasks = session.query(Task).all()

    if tasks:

        # Erstelle pd.DataFrame
        task_data = pd.DataFrame([
            {
                'task_id': task.id,
                'user-nachname': task.user.lastname if task.user else 'N/A',
                'user_id': task.user_id if task.user else 'N/A',
                'Version': task.version.name if task.version else 'N/A',
                'Revision': task.version_history.revision if task.version_history else 'N/A',
                'Start': task.start_timestamp,
                'Ende': task.end_timestamp,
                'Zeit': task.time,
                'Wahrgenommener Stress': task.perceived_stress,
                'Wahrgenommener Zeitdruck': task.perceived_time_pressure,
                'Wahrgenommene Frustration': task.perceived_frustration,
                'Wahrgenommene Komplexität': task.perceived_complexity,
                'Spielmodus': task.game_mode,
                'Tatsächliche Komplexität': task.version.complexity if task.version else 'N/A'
            }
            for task in tasks
        ])
        return task_data
    else:
        return pd.DataFrame(columns=['task_id', 'user-nachname', 'Version', 'Start', 'Ende', 'Wahrgenommene Komplexität', 'Tatsächliche Komplexität'])


def load_task_steps(task_id):
    """
    Lade alle Task-Steps (Montageschritte) aus der lokalen Datenbank

    Args:
       task_id: Aufgaben-ID, von welcher die Schritte geladen werden sollen

    Returns:
       pd.DataFrame: Alle Aufgabenschritte für die Aufgabe
    """
    # Ziehen aller Task Steps
    steps = session.query(TaskStep).filter_by(task_id=task_id).all()
    if steps:
        # Erstelle pd.DataFrame
        steps_data = pd.DataFrame([
            {
                'Schritt_Nr': step.step_number,
                'Start': step.start_time,
                'Ende': step.end_time,
                'Zeit': step.time_spent
            }
            for step in steps
        ])
        return steps_data
    else:
        return pd.DataFrame(columns=['Schritt_Nr', 'Start', 'Ende', 'Zeit'])


def get_users():
    """Lädt alle Nutzer aus der Datenbank. Returns: Alle Nutzer aus der Datenbank"""
    users = session.query(User).all()
    return users


def update_user(user_id, firstname, lastname, age, skill, nickname, selected_task_profiles):
    """Aktualisiert die Nutzerdaten in der Datenbank."""
    user = session.query(User).filter_by(id=user_id).first()
    if user:
        user.firstname = firstname
        user.lastname = lastname
        user.age = age
        user.skill = skill
        user.nickname = nickname

        user.task_profiles = [session.query(TaskProfile).filter_by(id=tp_id).first() for tp_id in selected_task_profiles]

        session.commit()


def is_nickname_unique(nickname, user_id):
    """Überprüft, ob der Nickname einzigartig ist. Returns: BOOL"""
    existing_user = session.query(User).filter(User.nickname == nickname, User.id != user_id).first()
    return existing_user is None


def is_version_unique(version_name):
    """Überprüft, ob eine Version einzigartig ist. Returns: BOOL"""
    existing_version = session.query(Version).filter(Version.name == version_name).first()

    return existing_version is None


def is_component_unique(component_name):
    """Überprüft, ob ein Bauteil einzigartig ist. Returns: BOOL"""
    existing_component = session.query(Component).filter(Component.name == component_name).first()

    return existing_component is None

def get_task_profiles():
    """Lädt alle TaskProfiles aus der Datenbank. Returns: Alle Aufgabenprofile ais der Datenbank"""
    task_profiles = session.query(TaskProfile).all()
    return task_profiles


def get_versions():
    """Lädt alle Versions aus der Datenbank. Returns: Alle Versionen ais der Datenbank"""
    versions = session.query(Version).all()
    return versions


def update_task_profile(task_profile_id, name, selected_versions_counts):
    """Aktualisiert die TaskProfile in der Datenbank."""
    task_profile = session.query(TaskProfile).filter_by(id=task_profile_id).first()
    if task_profile:
        task_profile.name = name

        # Lösche bestehende required_counts
        session.query(TaskProfileRequiredCount).filter_by(task_profile_id=task_profile_id).delete()

        # Aktualisiere die Versions des TaskProfiles und füge neue required_counts hinzu
        task_profile.versions = []
        for version_id, count in selected_versions_counts:
            version = session.query(Version).filter_by(id=version_id).first()
            if version:
                task_profile.versions.append(version)
                required_count = TaskProfileRequiredCount(task_profile_id=task_profile_id, version_id=version_id, count=count)
                session.add(required_count)

        session.commit()


# Funktion zum Löschen der gefilterten Tasks
def delete_filtered_tasks(task_ids):
    """Funktion zum Löschen der angegebenen Task IDs"""
    try:
        # Lösche die Tasks, die den gegebenen IDs entsprechen
        tasks_to_delete = session.query(Task).filter(Task.id.in_(task_ids)).all()
        for task in tasks_to_delete:
            session.delete(task)
        session.commit()
        return True, None
    except SQLAlchemyError as e:
        session.rollback()
        return False, str(e)


# Funktion zum Hinzufügen eines Task-Profils zu Nutzern
def add_task_profile_to_users(user_ids, task_profile_name):
    """Funktion zum Hinzufügen eines Task-Profils zu Nutzern"""
    try:
        # Hole das Task-Profil aus der Datenbank
        task_profile = session.query(TaskProfile).filter_by(name=task_profile_name).first()

        # Falls Task-Profil nicht existiert, gebe einen Fehler aus
        if not task_profile:
            return False, f"Task-Profile '{task_profile_name}' konnte nicht in der Datenbank gefunden werden"

        # Iteriere durch über User und füge das Task-Profil hinzu, falls es noch nicht zugewiesen ist
        for user_id in user_ids:
            user = session.query(User).get(user_id)
            if user and task_profile not in user.task_profiles:
                user.task_profiles.append(task_profile)

        session.commit()
        return True, "Task-Profile erfolgreich zugewiesen."

    except Exception as e:
        session.rollback()
        return False, str(e)


def toggle_task_profile_status(profile_id, active):
    """Funktion zum Wechseln des task_profil Status"""
    task_profile = session.query(TaskProfile).filter_by(id=profile_id).first()
    if task_profile:
        task_profile.active = active
        session.commit()


def toggle_version_status(version_id, active):
    """Funktion zum Wechseln des Versionsstatus"""
    version = session.query(Version).filter_by(id=version_id).first()
    if version:
        version.active = active
        session.commit()


# Funktion zur Berechnung der durchschnittlichen Bearbeitungszeit pro Version
def average_task_completion_time_by_version():

    """Funktion zur Berechnung der durchschnittlichen Bearbeitungszeit pro Version"""
    try:

        result = (
            session.query(
                Version.name,
                func.avg(Task.time)
            )
            .join(Task, Task.version_id == Version.id)
            .filter(Task.end_timestamp.isnot(None))
            .group_by(Version.name)
            .order_by(Version.name.desc())
            .all()
        )

        return {version_id: avg_time for version_id, avg_time in result}

    except SQLAlchemyError as e:
        print("Error calculating average task completion time by version:", e)
        return None


# Weitere Funktion zum Suchen der Aufgaben eines Nutzers -> Jedoch detaillierter als oben
def get_user_task_details(user_id):

    task_details = (
        session.query(Version.name, Version.complexity, Task.id, Task.time)
        .join(Task)
        .join(TaskStep, Task.id == TaskStep.task_id)
        .filter(Task.user_id == user_id)
        .group_by(Version.name, Task.id)
        .order_by(Version.name, Task.id)
        .all()
    )

    task_data = {}
    for version_name, version_complexity, task_id, task_time in task_details:
        if version_name not in task_data:
            task_data[version_name] = []

        task_data[version_name].append({
            'complexity': version_complexity,
            'task_id': task_id,
            'total_time': task_time,
            'steps': session.query(TaskStep.step_number, TaskStep.time_spent)
            .filter(TaskStep.task_id == task_id)
            .order_by(TaskStep.step_number)
            .all()
        })

    return task_data


# Funktion zum Speichern oder Aktualisieren einer Einstellung
def save_setting(key, value):
    """Funktion zum Speichern oder Aktualisieren einer Einstellung"""
    setting = session.query(Setting).filter_by(key=key).first()
    if setting:
        setting.value = value
    else:
        setting = Setting(key=key, value=value)
        session.add(setting)
    session.commit()

def get_setting(key, default=None):
    """Funktion zum Erhalten einer Einstellungs-Werts"""
    setting = session.query(Setting).filter_by(key=key).first()
    return setting.value if setting else default


def create_new_component(name, image_path):
    """
    Funktion zm Erstellen eines neuen Bauteil-Objekts

    Args:
       name: Name des Bauteils,
       image_path: Bildpfad für das Bauteil,

    Returns:
       Component-Objekt
    """
    # Neue Componente anlegen
    new_component = Component(
        name=name,
        component_image_path =image_path
    )

    session.add(new_component)
    session.commit()

    return new_component


def delete_all_components():
    """Funktion zum Löschen aller Bauteile"""
    session.query(Component).delete()
    session.commit()


def delete_component(component_id):
    """Funktion zum Löschen eines Bauteils"""
    component = session.query(Component).filter_by(id=component_id).first()

    if component:

        # Löschen der Einträge in component_list_required_counts
        session.query(ComponentListRequiredCount).filter_by(component_id=component_id).delete()

        # Löschen der Einträge in der Many-to-Many-Tabelle component_list_components
        session.execute(
            component_list_components_association.delete().where(
                component_list_components_association.c.component_id == component_id
            )
        )

        # Löschen der Komponente selbst
        session.delete(component)
        session.commit()


def update_component(component_id, new_name):
    """Funktion zum Aktualisieren eines Bauteilnamens"""
    component = session.query(Component).filter_by(id=component_id).first()
    if component:
        component.name = new_name
        session.commit()
        
             
# Gesamte Bauliste einer Version abrufen
def get_total_component_requirements_for_version(version_id):
    """Gesamte Bauliste einer Version abrufen"""
    component_counts = session.query(ComponentListRequiredCount).join(ComponentList).filter(ComponentList.version_id == version_id).all()
    total_requirements = {}
    for component in component_counts:
        total_requirements[component.component_id] = component.count
    return total_requirements


# Summiere die Bauteile aller Schritte einer Version
def get_total_assigned_components_for_version(version_id):
    """Summiere die Bauteile aller Schritte einer Version: Returns: assigned_counts """
    assigned_components = session.query(TaskComponentRequirement).join(Image).filter(Image.version_id == version_id).all()
    assigned_counts = {}
    for component in assigned_components:
        if component.component_id in assigned_counts:
            assigned_counts[component.component_id] += component.count
        else:
            assigned_counts[component.component_id] = component.count
    return assigned_counts


# Abgleich der gesamten Bauteilliste und der Bauteile der Schritte
def check_component_discrepancies(version_id):
    """Abgleich der gesamten Bauteilliste und der Bauteile der Schritte (Prüfe auf Diskrepanz)"""
    total_requirements = get_total_component_requirements_for_version(version_id)
    assigned_counts = get_total_assigned_components_for_version(version_id)

    discrepancies = []

    # Überprüfen, ob es Diskrepanzen bei den erforderlichen Bauteilen gibt
    for component_id, required_count in total_requirements.items():
        assigned_count = assigned_counts.get(component_id, 0)
        if assigned_count != required_count:
            component_name = session.query(Component).get(component_id).name
            discrepancies.append({
                "Bauteil": component_name,
                "Erforderlich": required_count,
                "Zugewiesen": assigned_count,
                "Abweichung": assigned_count - required_count
            })

    # Prüfen, ob es Bauteile gibt, die zugewiesen wurden, aber nicht in der Bauliste sind
    for component_id, assigned_count in assigned_counts.items():
        if component_id not in total_requirements:
            component_name = session.query(Component).get(component_id).name
            discrepancies.append({
                "Bauteil": component_name,
                "Erforderlich": 0,
                "Zugewiesen": assigned_count,
                "Abweichung": assigned_count
            })

    # Zeige die Diskrepanzen als Tabelle an
    if discrepancies:
        df_discrepancies = pd.DataFrame(discrepancies)
        st.write("Abweichungen zwischen der Bauliste und den zugewiesenen Bauteilen in den Schritten:")
        st.dataframe(df_discrepancies, use_container_width=True)
    else:
        st.success("Die Zuweisungen sind konsistent mit der Bauliste der Version.")

def get_step_status(self, r, frame):
    """Berechnet für ALLE Module des aktuellen Schritts: confidence + recognized."""
    if self.current_step == 0:
        return [{"label": "box", "recognized": self.box_live is not None, "overlap": 1.0, "confidence": 1.0}]
    
    step_index = self.current_step - 1
    step = self.module_layouts[self.active_variant][step_index]
    box = self.box_locked if self.box_is_locked else self.box_live
    
    status = []
    for item in step.get("items", []):
        label = item["label"]
        zone = self._zone_for_item(item, box)
        
        recognized = False
        overlap = 0.0
        confidence = 0.0
        
        if zone:
            det = self._best_det_for_zone(r, label, zone)
            if det:
                det_xyxy, conf, tid, ratio = det
                min_ov = float(item.get("min_overlap", step.get("min_overlap", self.DEFAULT_MIN_OVERLAP)))
                recognized = ratio >= min_ov
                overlap = round(ratio, 3)
                confidence = round(conf, 3)
        
        status.append({
            "label": label,
            "recognized": recognized,
            "overlap": overlap,
            "confidence": confidence
        })
    
    return status

def save_recognized_modules_status(self, task_id, session): 
    """Speichert den Status der Module."""
    step_status = getattr(self, '_last_status', []) 
    
    step = session.query(TaskStep).filter(
        TaskStep.task_id == task_id,
        TaskStep.step_number == self.current_step
    ).first()
    
    if step:
        step.recognition_status = json.dumps(step_status)
        step.recognized_modules = sum(1 for s in step_status if s["recognized"])
        step.total_modules = len(step_status)
        step.is_step_complete = all(s["recognized"] for s in step_status)
        
        session.commit()

