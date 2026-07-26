"""
DataMind AI — Multi-Agent CSV Data Analyst
--------------------------------------------------------------
A professional Streamlit front-end for the Supervisor / Agents
architecture defined in Supervisor.py, agents.py and tools.py.

Requirements:
    pip install streamlit pandas matplotlib seaborn langchain langchain-mistralai python-dotenv

Run:
    streamlit run app.py

Make sure a .env file with MISTRAL_API_KEY=<your_key> exists in the
same folder (Supervisor.py loads it via load_dotenv()).
--------------------------------------------------------------
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import tools  # noqa: E402  (module-level `df` lives here, shared by every tool)
from Supervisor import Supervisor, llm  # noqa: E402


# =================================================================
# PAGE CONFIG
# =================================================================
st.set_page_config(
    page_title="DataMind AI | Multi-Agent Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

AGENT_META = {
    "dataset_management": {"label": "📁 Dataset Management", "color": "#2563EB"},
    "eda":                {"label": "🔍 EDA Agent",           "color": "#7C3AED"},
    "visualization":      {"label": "📈 Visualization Agent",  "color": "#059669"},
    "data_cleaning":      {"label": "🧹 Data Cleaning Agent",  "color": "#D97706"},
    "general":            {"label": "🤖 General",              "color": "#6B7280"},
}


# =================================================================
# CUSTOM CSS — professional navy / white theme
# =================================================================
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F8FAFC;
        }
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #E2E8F0 !important;
            width: 100%;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            border-color: #3B82F6;
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
            background-color: #1E293B;
            border: 1px dashed #334155;
        }
        h1, h2, h3 {
            color: #0F172A;
        }
        div[data-testid="stChatMessage"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.5rem 0.9rem;
            margin-bottom: 0.4rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] div,
        div[data-testid="stChatMessage"] span:not(.agent-badge),
        div[data-testid="stChatMessage"] strong,
        div[data-testid="stChatMessage"] code {
            color: #0F172A !important;
        }
        .agent-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 6px;
        }
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 10px 12px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
            color: #64748B !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #0F172A !important;
        }
        /* Sidebar text that sits on white cards (metrics, dataframes) must stay dark */
        section[data-testid="stSidebar"] div[data-testid="stMetric"] * {
            color: #0F172A !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
            color: #64748B !important;
        }
        .app-header {
            padding: 1.1rem 1.4rem;
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
            border-radius: 14px;
            color: white;
            margin-bottom: 1rem;
        }
        .app-header h1 {
            color: #FFFFFF !important;
            margin-bottom: 0.15rem;
            font-size: 1.6rem;
        }
        .app-header p {
            color: #CBD5E1;
            margin: 0;
            font-size: 0.92rem;
        }
        /* The fixed footer that wraps st.chat_input — blend it into the page.
           Streamlit renders this container outside .stApp's own background,
           so html/body must be covered too, plus every known testid variant. */
        html, body {
            background-color: #F8FAFC !important;
        }
        div[data-testid="stBottom"],
        div[data-testid="stBottom"] > div,
        div[data-testid="stBottomBlockContainer"],
        div[data-testid="stAppViewContainer"],
        .stChatFloatingInputContainer,
        footer {
            background-color: #F8FAFC !important;
        }
        div[data-testid="stChatInput"] {
            background-color: #F8FAFC !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =================================================================
# SESSION STATE
# =================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "filename" not in st.session_state:
    st.session_state.filename = None


# =================================================================
# CACHED AGENTS (created once per session, reused across reruns)
# =================================================================
@st.cache_resource(show_spinner=False)
def load_agents():
    supervisor = Supervisor()
    agent_map = {
        "dataset_management": supervisor.dataset_management(llm),
        "eda": supervisor.eda(llm),
        "visualization": supervisor.visualization(llm),
        "data_cleaning": supervisor.data_cleaning(llm),
    }
    return supervisor, agent_map


supervisor, agent_map = load_agents()


# =================================================================
# CORE QUERY HANDLER
# =================================================================
def process_query(query: str):
    """Route a user query through the Supervisor, run the chosen agent,
    capture any matplotlib figures it produced, and store everything
    in session_state for rendering."""

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Analyzing your request..."):
        try:
            decision = supervisor.create_decision(query)
            route = getattr(decision, "content", str(decision)).strip().lower()
            if route not in agent_map:
                route = "general"

            if route == "general":
                # Not a dataset task (e.g. "hi", "thanks", small talk) —
                # answer naturally with the plain LLM instead of forcing
                # it through a specialized agent. Works even with no CSV
                # uploaded yet.
                chit_chat_prompt = f"""
You are the friendly assistant embedded inside "DataMind AI", a CSV
data-analysis app. The user's message does not require any of the
data agents (dataset preview, EDA, visualization, cleaning).

Reply briefly and naturally, like a normal helpful assistant. If it
makes sense, you may mention in passing that you can also analyze
their uploaded CSV, but do not force it or repeat this every time.

User message: {query}
"""
                reply = llm.invoke(chit_chat_prompt)
                answer = getattr(reply, "content", str(reply)) or "Hi! How can I help?"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "agent": "general",
                        "content": answer,
                        "figures": [],
                    }
                )
                return

            if not st.session_state.dataset_loaded:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "agent": route,
                        "content": "⚠️ That needs a dataset — please upload a CSV file from the sidebar first, then ask again.",
                        "figures": [],
                    }
                )
                return

            agent = agent_map[route]

            # The agent gets a fresh, stateless call each time — it has no
            # memory of the Streamlit session. Since the CSV is loaded
            # directly into tools.df by the sidebar uploader (not via the
            # CSV_loader tool), we tell the agent explicitly that data is
            # already available so it doesn't ask the user to upload again.
            augmented_query = (
                "[Context: A CSV dataset is already loaded and ready in memory. "
                "Do not call CSV_loader or ask the user to upload a file — just "
                "use the appropriate tool to answer directly.]\n\n"
                f"User question: {query}"
            )

            # Track figures already open so we only pick up NEW ones from this call
            figs_before = set(plt.get_fignums())

            result = agent.invoke({"messages": [{"role": "user", "content": augmented_query}]})
            final_message = result["messages"][-1]
            answer = getattr(final_message, "content", str(final_message)) or "Done."

            new_fig_nums = set(plt.get_fignums()) - figs_before
            figures = [plt.figure(num) for num in new_fig_nums]

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "agent": route,
                    "content": answer,
                    "figures": figures,
                }
            )

            # Free matplotlib's global state now that figures are captured
            for num in new_fig_nums:
                plt.close(num)

        except Exception as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "agent": "general",
                    "content": f"❌ Something went wrong while processing that request: `{e}`",
                    "figures": [],
                }
            )


# =================================================================
# SIDEBAR
# =================================================================
with st.sidebar:
    st.markdown("## 📊 DataMind AI")
    st.caption("Multi-Agent CSV Data Analyst")
    st.divider()

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
        try:
            tools.df = pd.read_csv(uploaded_file)
            st.session_state.dataset_loaded = True
            st.session_state.filename = uploaded_file.name
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "agent": "dataset_management",
                    "content": (
                        f"✅ **{uploaded_file.name}** loaded successfully — "
                        f"**{tools.df.shape[0]} rows × {tools.df.shape[1]} columns**.\n\n"
                        "Ask me anything about it, or use the quick actions below."
                    ),
                    "figures": [],
                }
            )
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

    if st.session_state.dataset_loaded and tools.df is not None:
        st.success(f"📄 {st.session_state.filename}")


        st.divider()
        st.markdown("#### ⚡ Quick Actions")

        if st.button("👀  Preview rows"):
            process_query("Show me the first and last 5 rows of the dataset")
        if st.button("📋  Dataset info"):
            process_query("Give me dataset info including shape, columns and dtypes")
        if st.button("🔎  Full EDA summary"):
            process_query("Give me a full exploratory data analysis summary of the dataset")
        if st.button("🚨  Missing values"):
            process_query("Show missing value percentage for each column")
        if st.button("🔥  Correlation heatmap"):
            process_query("Generate a correlation heatmap for the numeric columns")
        if st.button("🧹  Auto-clean summary"):
            process_query("Generate a data cleaning summary report of the dataset")

        st.divider()
        if os.path.exists("Cleaned_dataset.csv"):
            with open("Cleaned_dataset.csv", "rb") as f:
                st.download_button(
                    "⬇️  Download cleaned CSV",
                    f,
                    file_name="Cleaned_dataset.csv",
                    mime="text/csv",
                )

    st.divider()
    if st.button("🗑️  Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
A **Supervisor Agent** reads every query and routes it to one of
four specialized agents:

- 📁 **Dataset Management** — load & preview data
- 🔍 **EDA Agent** — stats, missing values, correlation, filtering
- 📈 **Visualization Agent** — charts & plots
- 🧹 **Data Cleaning Agent** — fill nulls, dedupe, rename, save

Built with **LangChain**, **Mistral AI**, and **Streamlit**.
"""
        )


# =================================================================
# MAIN AREA
# =================================================================
st.markdown(
    """
    <div class="app-header">
        <h1>🧠 DataMind AI</h1>
        <p>A multi-agent conversational assistant for CSV data analysis, cleaning & visualization</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_preview = st.tabs(["💬 Chat Assistant", "📄 Data Preview"])

# --------------------------- CHAT TAB ---------------------------
with tab_chat:
    if not st.session_state.dataset_loaded:
        st.info("👈 Upload a CSV file from the sidebar to get started.")

    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and msg.get("agent"):
                meta = AGENT_META.get(msg["agent"], AGENT_META["general"])
                st.markdown(
                    f"<span class='agent-badge' style='background:{meta['color']}1F;"
                    f"color:{meta['color']};'>{meta['label']}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            for fig in msg.get("figures", []):
                st.pyplot(fig, use_container_width=True)

# ------------------------- PREVIEW TAB ---------------------------
with tab_preview:
    if st.session_state.dataset_loaded and tools.df is not None:
        st.markdown(f"#### {st.session_state.filename}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{tools.df.shape[0]:,}")
        m2.metric("Columns", tools.df.shape[1])
        m3.metric("Missing %", f"{tools.df.isnull().mean().mean() * 100:.1f}%")
        m4.metric("Duplicate rows", int(tools.df.duplicated().sum()))

        st.markdown("##### Preview")
        st.dataframe(tools.df.head(50), use_container_width=True)

        st.markdown("##### Column Types")
        dtypes_df = pd.DataFrame(
            {"Column": tools.df.columns, "Data Type": tools.df.dtypes.astype(str).values}
        )
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
    else:
        st.info("Upload a CSV from the sidebar to see the data preview here.")

# ------------------- CHAT INPUT (pinned to bottom) -------------------
# Placed outside the tabs on purpose: st.chat_input only docks to the
# bottom of the page when it isn't nested inside a container/tab.
query = st.chat_input("Ask about your dataset... e.g. 'Show a histogram of Sales'")
if query:
    process_query(query)
    st.rerun()