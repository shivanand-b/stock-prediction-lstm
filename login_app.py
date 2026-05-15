import streamlit as st
import sqlite3
from streamlit_option_menu import option_menu
# 🔐 SESSION STATE INIT
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# PAGE CONFIG
st.set_page_config(
    page_title="AI Stock Prediction System",
    layout="wide"
)

# DATABASE CONNECTION
conn = sqlite3.connect(
    'users.db',
    check_same_thread=False
)

cursor = conn.cursor()

# CREATE TABLE
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT,
    password TEXT
)
""")

conn.commit()

# FUNCTIONS

# ADD USER
def add_user(username, password):

    cursor.execute(
        "INSERT INTO users VALUES (?, ?)",
        (username, password)
    )

    conn.commit()

# LOGIN USER
def login_user(username, password):

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    data = cursor.fetchone()

    return data

# TITLE
st.title("📈 AI Stock Prediction System")

# MENU
selected = option_menu(
    menu_title=None,
    options=["Login", "Signup"],
    icons=["box-arrow-in-right", "person-plus"],
    orientation="horizontal"
)

# LOGIN PAGE
if selected == "Login":

    st.subheader("Login Account")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        result = login_user(
            username,
            password
        )

        if result:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username} 🔥")
            st.switch_page("pages/dashboard.py")

# SIGNUP PAGE
if selected == "Signup":

    st.subheader("Create New Account")

    new_user = st.text_input(
        "Create Username"
    )

    new_password = st.text_input(
        "Create Password",
        type="password"
    )

    if st.button("Signup"):

        add_user(
            new_user,
            new_password
        )

        st.success(
            "Account Created Successfully ✅"
        )

        st.info(
            "Go to Login Menu to Login"
        )
