"""Render the employee question and source view."""

import logging
from pathlib import Path

import streamlit as st

from nool.app.access import citation_is_current
from nool.data.documents import source_digest
from nool.retrieval.company import DEMO_USERS

ROLE_LABELS = {
    "employee": "Employee",
    "hr_admin": "People Operations",
    "it_admin": "IT",
}


def show_sources(citations, user_id: str, service, message_index: int) -> None:
    if not citations:
        return
    with st.expander(f"Sources · {len(citations)}"):
        for index, citation in enumerate(citations):
            if not citation_is_current(citation, user_id, service):
                continue
            document = service.store.get_document(citation.document_id)
            st.markdown(f"**{citation.title}**")
            st.caption(
                f"{citation.document_id} · {citation.section} · v{citation.version}"
            )
            st.write(f"“{citation.quote}”")
            source = Path(document.source_path)
            if source.is_file() and source_digest(source) == document.content_hash:
                st.download_button(
                    "View document",
                    source.read_bytes(),
                    file_name=f"{document.document_id}{source.suffix.lower()}",
                    key=f"source_{message_index}_{index}",
                )
            if index < len(citations) - 1:
                st.divider()


def render_workspace(service, email: str, user_id: str, logout=None) -> None:
    with st.sidebar:
        st.markdown("### Nool")
        st.caption("Company knowledge")
        st.divider()
        st.caption("Signed in as")
        st.write(email)
        st.caption(ROLE_LABELS[DEMO_USERS[user_id]])
        if logout is not None:
            st.button("Sign out", on_click=logout, use_container_width=True)

    st.title("Ask Nool")
    st.caption("Find answers in company policies and guides.")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="Ask about a company policy or guide",
            max_chars=500,
        )
        submitted = st.form_submit_button("Ask question", type="primary")

    identity = (email.strip().casefold(), user_id)
    if st.session_state.get("identity") != identity:
        st.session_state.identity = identity
        st.session_state.messages = []

    for message_index, message in enumerate(st.session_state.messages):
        with st.container(border=True):
            st.caption("QUESTION" if message["role"] == "user" else "ANSWER")
            citations = message.get("citations", ())
            if citations and not all(
                citation_is_current(citation, user_id, service)
                for citation in citations
            ):
                st.write("This answer is no longer available. Please ask again.")
            else:
                st.write(message["text"])
                if message["role"] == "assistant":
                    show_sources(citations, user_id, service, message_index)

    if submitted and question.strip():
        st.session_state.messages.append({"role": "user", "text": question})
        with st.spinner("Checking company documents..."):
            try:
                result = service.answer(question, user_id)
                reply = {
                    "role": "assistant",
                    "text": result.text,
                    "citations": result.citations,
                }
            except Exception:
                logging.exception("Nool could not answer a question")
                reply = {
                    "role": "assistant",
                    "text": (
                        "I couldn't complete the document check right now. "
                        "Please try again shortly."
                    ),
                    "citations": (),
                }
        st.session_state.messages.append(reply)
        st.rerun()
