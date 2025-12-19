import streamlit as st

from data.database_functions import get_users, is_nickname_unique, update_user, get_task_profiles
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




st.title("Benutzerverwaltung")

# Lade Nutzerinformationen
users = get_users()
user_ids = [user.id for user in users]
user_dict = {user.id: user for user in users}

# Lade Aufgabenprofile
task_profiles = get_task_profiles()
task_profile_dict = {tp.id: tp for tp in task_profiles}


def user_management():

    # Auswahlbox über alle Nutzer
    selected_user_id = st.selectbox("Wähle einen Nutzer aus", user_ids, format_func=lambda x: user_dict[x].nickname)

    if selected_user_id:
        selected_user = user_dict[selected_user_id]

        st.subheader("Nutzerdaten bearbeiten")

        # Textinputs zur Bearbeitung der Nutzerdaten
        firstname = st.text_input("Vorname", value=selected_user.firstname)
        lastname = st.text_input("Nachname", value=selected_user.lastname)
        nickname = st.text_input("Nickname", value=selected_user.nickname)
        age = st.number_input("Alter", value=selected_user.age, min_value=0, step=1)
        skill = st.text_input("Skill", value=selected_user.skill)


        # Mehrfachauswahlbox für TaskProfiles
        selected_task_profiles = st.multiselect(
            "TaskProfiles",
            [tp.id for tp in task_profiles],
            default=[tp.id for tp in selected_user.task_profiles], # Die beim Nutzer hinterlegten task_profiles als default
            format_func=lambda x: task_profile_dict[x].name
        )

        # Speichern der Änderungen
        if st.button("Änderungen speichern"):

            # Prüfen, dass Nickname einzigartig
            if is_nickname_unique(nickname, selected_user_id):
                update_user(selected_user_id, firstname, lastname, age, skill, nickname, selected_task_profiles)
                st.success("Nutzerdaten wurden erfolgreich aktualisiert.")
            else:
                st.error("Der Nickname ist bereits vergeben. Bitte wähle einen anderen Nickname.")


user_management()