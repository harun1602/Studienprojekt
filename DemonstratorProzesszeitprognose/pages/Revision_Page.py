from data.database_code import session, VersionHistory, ImageHistory, Version
import streamlit as st
from navigation import make_sidebar

st.set_page_config(initial_sidebar_state="expanded", layout="centered")

# Skript für die Revisionsseite

def get_version_revisions(version_id):

    # Abrufen aller Revisionen für die gegebene Version
    revisions = session.query(VersionHistory).filter_by(version_id=version_id).order_by(VersionHistory.revision.desc()).all()

    if not revisions:
        print(f"Keine Revisionen für Version mit ID {version_id} gefunden.")
        return []

    return revisions


def get_images_of_revision(version_history_id):

    # Alle Bilder der bestimmten Revision abrufen
    images = session.query(ImageHistory).filter_by(version_history_id=version_history_id).all()

    if not images:
        st.warning(f"Keine Bilder für Revision {version_history_id} gefunden.")
        return []

    return images


def show_images_of_revision(version_id, revision_number):

    # Abrufen der gewünschten Revision
    revision = session.query(VersionHistory).filter_by(version_id=version_id, revision=revision_number).first()

    if not revision:
        st.warning(f"Keine Revision {revision_number} für Version {version_id} gefunden.")
        return

    # Abrufen der Bilder dieser Revision
    images = get_images_of_revision(revision.id)

    if not images:
        st.warning(f"Keine Bilder für Revision {revision_number} gefunden.")
        return

    # Bilder der Revision anzeigen
    st.write(f"Bilder für Revision {revision_number}:")
    for image in images:
        st.write(f"Bild: {image.image_path}")
        st.write(f"Anleitung: {image.image_anleitung}")
        st.image(image.image_path, width='stretch')

        st.divider()



def show_revision():

    make_sidebar()

    versions = session.query(Version).all()

    # Liste über alle Versionsnamen
    version_names = [version.name for version in versions]

    # Selectbox zum Auswählen der gewünscten Version
    selected_version_name = st.selectbox("Wählen Sie eine Version zur Bearbeitung", version_names)

    if selected_version_name:

        selected_version = session.query(Version).filter_by(name=selected_version_name).first()

        version_revisions = get_version_revisions(selected_version.id)

        revision_ids = [revision.revision for revision in version_revisions]

        selected_version_revision_id = st.selectbox("Wahlen Sie eine Revision der Version", revision_ids)

        # Anzeige der Revision-Daten der Version
        if selected_version_revision_id:

            selected_version_revision = session.query(VersionHistory).filter_by(version_id=selected_version.id,revision=selected_version_revision_id).first()

            st.write(f"Revision {selected_version_revision.revision} - Name: {selected_version_revision.name}")
            st.write(f"Beschreibung: {selected_version_revision.description}")
            st.write(f"Komplexität: {selected_version_revision.complexity}")
            st.write(f"Zeitlimit: {selected_version_revision.time_limit}")
            st.write(f"Erstellt am: {selected_version_revision.timestamp}")

            # Zeige Images und Anleitungen
            show_images_of_revision(selected_version.id, selected_version_revision.revision)


show_revision()
