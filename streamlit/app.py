import streamlit as st
import requests
import json
import re
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080")
REQUEST_TIMEOUT = 180  # seconds

st.set_page_config(page_title="CouncilAI", page_icon="📚", layout="wide")

_DEFAULTS = {
    "chat_history": [],
    "explain_history": [],
    "questions_history": [],
    "token": None,
    "user_id": None,
    "doc_id": None,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val
def api_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}
def meta_caption(data: dict) -> str:
    parts = [
        f"Confidence: {data.get('confidence', 0):.0%}",
        f"Source: {data.get('source', 'unknown')}",
        f"Latency: {data.get('latency', '?')}",
    ]
    if data.get("cache_hit"):
        parts.append("⚡ Cache hit")
    if data.get("peer_reviewed"):
        parts.append("📝 Peer reviewed")
    return " · ".join(parts)
def try_parse_questions(raw: str) -> list:
    try:
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw)
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    return []
def login_section():
    st.sidebar.header("🔐 Authentication")
    tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/login",
                    json={"username": username, "password": password},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token = data["token"]
                    st.session_state.user_id = data["user_id"]
                    st.success(f"Welcome, {data['user_id']}!")
                    st.rerun()
                else:
                    st.error(resp.json().get("error", "Login failed"))
            except requests.ConnectionError:
                st.error("Cannot connect to backend. Is it running?")

    with tab_register:
        new_user = st.text_input("Username", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register"):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/register",
                    json={"username": new_user, "password": new_pass},
                )
                if resp.status_code == 201:
                    data = resp.json()
                    st.session_state.token = data["token"]
                    st.session_state.user_id = data["user_id"]
                    st.success("Account created!")
                    st.rerun()
                else:
                    st.error(resp.json().get("error", "Registration failed"))
            except requests.ConnectionError:
                st.error("Cannot connect to backend. Is it running?")
st.header("📚 CouncilAI")

if not st.session_state.token:
    login_section()
    st.info("Please log in to continue. Default: demo / demo123")
    st.stop()

# Sidebar: logged-in state
st.sidebar.markdown(f"**Logged in as:** {st.session_state.user_id}")
if st.sidebar.button("Logout"):
    for key in _DEFAULTS:
        st.session_state[key] = _DEFAULTS[key]
    st.rerun()

with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg"])

    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Ingesting..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/ingest",
                    headers=api_headers(),
                    files={"file": (uploaded_file.name, uploaded_file.read())},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.doc_id = data["doc_id"]
                    st.success(
                        f"✅ Ingested {data['chunk_count']} chunks "
                        f"(OCR: {data['metadata'].get('file_type', 'unknown')})"
                    )
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to backend")

    if st.session_state.doc_id:
        st.info(f"Active document: `{st.session_state.doc_id}`")

ask_tab, explain_tab, questions_tab = st.tabs(
    ["💬 Ask", "📖 Explain", "📝 Generate Questions"]
)

with ask_tab:
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))
            if msg.get("meta"):
                st.caption(msg["meta"])

    user_question = st.chat_input("Ask a question about the document")
    if user_question:
        if not st.session_state.doc_id:
            st.warning("Please upload and ingest a document first.")
        else:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            with st.spinner("Thinking... (Multi-LLM Council in action)"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/query",
                        headers=api_headers(),
                        json={
                            "question": user_question,
                            "doc_id": st.session_state.doc_id,
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "meta": meta_caption(data),
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Error: {resp.json().get('error', resp.text)}",
                        })
                except requests.ConnectionError:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "❌ Cannot connect to backend. Is it running?",
                    })
                except requests.Timeout:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "⏱️ Request timed out. The LLM Council may need more time.",
                    })
            st.rerun()

with explain_tab:
    if not st.session_state.doc_id:
        st.info("Upload and ingest a document first to generate explanations.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            knowledge_level = st.selectbox(
                "Knowledge Level",
                ["beginner", "intermediate", "advanced"],
                index=0,
                key="explain_level",
            )
        with col2:
            depth = st.selectbox(
                "Depth",
                ["brief", "section-wise", "detailed"],
                index=1,
                key="explain_depth",
            )

        focus_topics = st.text_input(
            "Focus Topics (optional, comma-separated)",
            key="explain_focus",
            placeholder="e.g. attention mechanism, transformers",
        )

        generate_explain = st.button(
            "🚀 Generate Explanation", key="explain_btn", use_container_width=True
        )

        if generate_explain:
            with st.spinner("Generating explanation via LLM Council..."):
                try:
                    payload = {
                        "doc_id": st.session_state.doc_id,
                        "knowledge_level": knowledge_level,
                        "depth": depth,
                    }
                    if focus_topics.strip():
                        payload["focus_topics"] = [
                            t.strip() for t in focus_topics.split(",") if t.strip()
                        ]
                    resp = requests.post(
                        f"{API_BASE}/api/v1/explain",
                        headers=api_headers(),
                        json=payload,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.explain_history.append({
                            "level": knowledge_level,
                            "depth": depth,
                            "explanation": data["explanation"],
                            "meta": meta_caption(data),
                        })
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.json().get('error', resp.text)}")
                except requests.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except requests.Timeout:
                    st.error("Request timed out. Try again.")

        # History (newest first)
        for i, entry in enumerate(reversed(st.session_state.explain_history)):
            idx = len(st.session_state.explain_history) - i
            with st.expander(
                f"Explanation #{idx} — {entry['level'].capitalize()}, {entry['depth']}",
                expanded=(i == 0),
            ):
                st.markdown(entry["explanation"])
                st.caption(entry["meta"])

        if st.session_state.explain_history:
            if st.button("🗑️ Clear Explanation History", key="clear_explain", type="secondary"):
                st.session_state.explain_history = []
                st.rerun()

with questions_tab:
    if not st.session_state.doc_id:
        st.info("Upload and ingest a document first to generate questions.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            num_questions = st.number_input(
                "Number of Questions", min_value=1, max_value=20, value=5, key="q_num"
            )
        with col2:
            difficulty = st.slider("Difficulty (1-10)", 1, 10, 5, key="q_diff")
        with col3:
            question_type = st.selectbox(
                "Type", ["subjective", "mcq"], index=0, key="q_type"
            )

        bloom_level = st.selectbox(
            "Bloom's Level (optional)",
            ["", "remember", "understand", "apply", "analyze", "evaluate", "create"],
            index=0,
            key="q_bloom",
        )

        generate_q = st.button(
            "🚀 Generate Questions", key="q_btn", use_container_width=True
        )

        if generate_q:
            with st.spinner("Generating questions via LLM Council..."):
                try:
                    payload = {
                        "doc_id": st.session_state.doc_id,
                        "num_questions": num_questions,
                        "difficulty": difficulty,
                        "question_type": question_type,
                    }
                    if bloom_level:
                        payload["bloom_level"] = bloom_level
                    resp = requests.post(
                        f"{API_BASE}/api/v1/generate-questions",
                        headers=api_headers(),
                        json=payload,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.questions_history.append({
                            "type": question_type,
                            "difficulty": difficulty,
                            "num": num_questions,
                            "questions": data.get("questions") or [],
                            "raw_output": data.get("raw_output", ""),
                            "meta": meta_caption(data),
                        })
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.json().get('error', resp.text)}")
                except requests.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except requests.Timeout:
                    st.error("Request timed out. Try again.")

        # History (newest first)
        for i, entry in enumerate(reversed(st.session_state.questions_history)):
            idx = len(st.session_state.questions_history) - i
            label = f"Set #{idx} — {entry['type'].upper()}, difficulty {entry['difficulty']}/10"

            with st.expander(label, expanded=(i == 0)):
                questions_list = entry.get("questions") or []

                # Fallback: try parsing raw_output on Python side
                if not questions_list and entry.get("raw_output"):
                    questions_list = try_parse_questions(entry["raw_output"])

                if questions_list:
                    for qi, q in enumerate(questions_list, 1):
                        st.markdown(f"**Q{qi}.** {q.get('question', '')}")

                        if q.get("options"):
                            for opt in q["options"]:
                                st.markdown(f"  - {opt}")

                        # Answer hidden behind a toggle
                        if st.checkbox(
                            f"Show answer for Q{qi}",
                            key=f"reveal_{idx}_{qi}",
                            value=False,
                        ):
                            st.success(f"**Answer:** {q.get('answer', 'N/A')}")
                            if q.get("explanation"):
                                st.info(f"**Explanation:** {q['explanation']}")

                        st.markdown("---")
                else:
                    st.markdown(entry.get("raw_output", "No questions generated."))

                st.caption(entry["meta"])

        if st.session_state.questions_history:
            if st.button("🗑️ Clear Questions History", key="clear_questions", type="secondary"):
                st.session_state.questions_history = []
                st.rerun()
