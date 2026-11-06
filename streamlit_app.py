"""Nool's employee question and answer website."""

import logging
from pathlib import Path
from tempfile import gettempdir

import streamlit as st

from nool.app.access import demo_user_for_email
from nool.app.page import render_workspace
from nool.app.runtime import load_answer_service

PROJECT_ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Nool | Company answers",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def answer_service():
    return load_answer_service(PROJECT_ROOT, Path(gettempdir()) / "nool")


def assigned_user() -> str | None:
    try:
        assignments = st.secrets["access"]
    except (KeyError, FileNotFoundError):
        return None
    email = st.user.get("email")
    return demo_user_for_email(email, assignments)


if not st.user.get("is_logged_in", False):
    st.title("Nool")
    st.write("Company answers, with sources you can check.")
    try:
        st.secrets["auth"]
    except (KeyError, FileNotFoundError):
        st.info("Sign-in is being set up. Please return later.")
    else:
        st.button("Sign in", type="primary", on_click=st.login)
    st.stop()

user_id = assigned_user()
if user_id is None:
    st.title("Access unavailable")
    st.write("Your account is not set up for this workspace.")
    st.button("Sign out", on_click=st.logout)
    st.stop()

try:
    service = answer_service()
except Exception:
    logging.exception("Nool answer service could not start")
    st.error("Answers are temporarily unavailable. Please try again later.")
    st.stop()

render_workspace(service, st.user.get("email", ""), user_id, st.logout)
