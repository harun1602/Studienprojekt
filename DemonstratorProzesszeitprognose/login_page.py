import streamlit as st
from data.database_functions import add_user, get_task_profiles
from data.database_code import User, Task, TaskProfile
from data.database_code import session


def login_form():

    st.set_page_config(
        page_title="Willkommen 👋",
        initial_sidebar_state="collapsed",
        layout="centered")


    st.write("# Herzlich Willkommen...")

    st.markdown(
        """
        ... in unserem Prozesszeitprognosen Demonstrator mit SL und Scheduling mit RL\n
        Bitte lege zuerst einen neuen Nutzer an, oder melde dich als ein vorhandenen Nutzer an

    """
    )

    st.session_state.current_user = None

    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None

    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    users = session.query(User).all()

    if st.checkbox('Vorhandenen Nutzer auswählen'):
        if users:
            user_list = [f"{user.firstname} {user.lastname} ({user.nickname})" for user in users]

            def check_if_Admin():
                if st.session_state.selected_user is not None:
                    selected_user_obj_admin = users[user_list.index(st.session_state.selected_user)]
                    if selected_user_obj_admin.firstname == 'ADMIN' and selected_user_obj_admin.lastname == 'ADMIN':
                        st.session_state.is_admin = True
                    else:
                        st.session_state.is_admin = False

            selected_user = st.selectbox("Wer bist du?", user_list, index=None, placeholder="Wähle einen Nutzer!", on_change=check_if_Admin, key='selected_user')

            if st.session_state.is_admin is True:
                admin_password = st.text_input("Password", type="password")

            if st.button("Nutzer wählen"):
                if selected_user is not None:
                    if st.session_state.is_admin is False:

                        selected_user_obj = users[user_list.index(st.session_state.selected_user)]

                        if selected_user_obj != st.session_state.current_user:

                            resets = ['completed_tasks', 'remaining_tasks', 'current_profile', 'game_mode']
                            # Delete all the items in Session state
                            for key in resets:
                                if key in st.session_state:
                                    del st.session_state[key]

                        st.session_state.current_user = selected_user_obj
                        st.session_state.logged_in = True
                        st.success(f"Eingeloggt als {selected_user_obj.firstname} {selected_user_obj.lastname}")
                        #sleep(2)
                        st.rerun()

                    elif st.session_state.is_admin and admin_password == "ADMIN":
                        selected_user_obj = users[user_list.index(st.session_state.selected_user)]
                        st.session_state.current_user = selected_user_obj
                        st.session_state.logged_in = True
                        st.success(f"Eingeloggt als {selected_user_obj.firstname} {selected_user_obj.lastname}")
                        st.rerun()

                    else:
                        st.error("Incorrect password for ADMIN")

        else:
            st.warning("Es sind keine Benutzer in der Datenbank vorhanden. Bitte legen Sie einen neuen Benutzer an.")

    else:

        # Lade Aufgabenprofile
        task_profiles = get_task_profiles()
        task_profile_dict = {tp.id: tp for tp in task_profiles}

        st.subheader("Neuen Nutzer anlegen")
        user_firstname = st.text_input('Vorname des Nutzers')
        user_lastname = st.text_input('Nachname des Nutzers')
        user_nickname = st.text_input('Nickname des Nutzers')
        user_age = st.number_input("Alter des Nutzers", min_value=0, max_value=100, step=1)
        user_skill = st.text_input("Skill des Nutzers")

        # Mehrfachauswahlbox für TaskProfiles
        selected_task_profiles = st.multiselect(
            "TaskProfiles",
            [tp.id for tp in task_profiles],
            format_func=lambda x: task_profile_dict[x].name
        )


        if st.button("Nutzer anlegen"):
            try:
                add_user(session, user_firstname, user_lastname, user_nickname, user_age, user_skill, selected_task_profiles)
                new_user = session.query(User).filter_by(nickname=user_nickname).first()
                st.session_state.current_user = new_user
                st.session_state.logged_in = True
                st.success(f"Neuer Nutzer {user_firstname} {user_lastname} erfolgreich angelegt")
                st.rerun()

            except ValueError as e:
                st.error(e)
                st.session_state.current_user = None
                st.session_state.logged_in = False
                st.rerun()



