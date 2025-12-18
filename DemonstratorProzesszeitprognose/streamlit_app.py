import streamlit as st
from login_page import login_form

# Es handelt sich hierbei um den Einstiegspunkt der Anwendung

# Falls eingeloggt:
# Wenn der Nutzer der Admin ist -> Wechseln der Page auf die Admin-Seite
# Sonst -> Wechseln der Page auf den Arbeitsplatz
if 'logged_in' in st.session_state and st.session_state.logged_in:
    if not st.session_state.get("is_admin", False):
        st.switch_page("pages/Arbeitsplatz.py")
    else:
        st.switch_page("pages/Admin_Tasks.py")
# Falls kein Nutzer eingeloggt ist -> login_form starten
else:
    login_form()
