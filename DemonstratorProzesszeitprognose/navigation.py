import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Dieses Skript regelt die Navigation in der Sidebar des UI

def get_current_page_name():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Couldn't get script context")

    pages = st.runtime.get_instance()._pages_manager.get_pages()

    return pages[ctx.page_script_hash]["page_name"]


def make_sidebar():
    with st.sidebar:
        st.title("Navigator")
        st.write("")
        st.write("")

        sidebar_logo = "images/LPS_Logo.png"
        main_body_logo = "images/LPS_Logo.png"

        st.logo(sidebar_logo, icon_image=main_body_logo)

        st.logo(
            sidebar_logo,
            link="https://www.lps.ruhr-uni-bochum.de",
            #icon_image=LOGO_URL_SMALL,
        )

        if st.session_state.get("logged_in", False):

            # Wenn man nicht als Admin eingeloggt ist
            if not st.session_state.get("is_admin", False):

                # Links zu den entsprechenden Skripten (Seiten), die ausgeführt werden sollen
                st.page_link("pages/Arbeitsplatz.py", label="Dein Arbeitplatz", icon="💼")
                st.page_link("pages/User_Statistics.py", label="Dein Arbeitsverlauf", icon="📖")


            # Wenn man als Admin eingeloggt ist
            else:

                # Links zu den entsprechenden Skripten (Seiten), die ausgeführt werden sollen
                st.page_link("pages/Admin_Tasks.py", label="Task Management", icon="📝")
                st.page_link("pages/Admin_Users.py", label="User Management", icon="👥")
                st.page_link("pages/Admin_Statistics.py", label="Global Statistics Page", icon="🌍")
                #st.page_link("pages/Admin_KI_Prognose_2.py", label="KI_Prognosen", icon="🤖")
                st.page_link("pages/Revision_Page.py", label="Revision-Page", icon="🕰️")


            st.write("")
            st.write("")

            if st.button("Log out"):
                logout()

        elif get_current_page_name() != "streamlit_app":
            st.switch_page("streamlit_app.py")

# Funktion zum Ausloggen
def logout():
    st.session_state.logged_in = False
    st.session_state.is_admin = False
    st.session_state.current_profile = None
    st.info("Erfolgreich ausgelogged")
    st.switch_page("streamlit_app.py")