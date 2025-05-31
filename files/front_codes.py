import streamlit as st
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import time
from files.backend_codes import build_workflow
import uuid # Add this import

compiled_workflow = build_workflow()

# --- 1. セッション状態で履歴＆state管理 ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "memory_state" not in st.session_state:
    st.session_state["memory_state"] = {}
if "disabled" not in st.session_state: # Initialization for chat input disable state
    st.session_state["disabled"] = False

# --- 2. ワークフローのセットアップ（必要ならキャッシュ） ---
MAX_HISTORY_DISPLAY = 20 # Max number of chat messages to display

@st.cache_resource
def get_workflow():
    return build_workflow()

compiled_workflow = get_workflow()

# --- Clear History Button ---
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.memory_state = {} # Clear local frontend state

    # Ensure a thread_id exists for the current session for clearing history
    if "session_thread_id" not in st.session_state:
        st.session_state["session_thread_id"] = str(uuid.uuid4())

    # Use the session-specific thread_id for clearing
    session_specific_thread_id = st.session_state["session_thread_id"]
    config = {'configurable': {'thread_id': session_specific_thread_id}}
    try:
        # Directly invoke the backend with the special clear command
        compiled_workflow.invoke({"input": "SYSTEM_CLEAR_HISTORY"}, config=config)
        st.success(f"Chat history and associated backend state for your session have been cleared.")
    except Exception as e:
        st.error(f"Error clearing backend state for session {session_specific_thread_id}: {e}")
    st.rerun() # Rerun to refresh the UI cleanly after clearing

# --- 3. チャットUI ---
st.title("SQL生成&データ解釈チャットAI")

# 3.1 履歴の表示（直近MAX_HISTORY_DISPLAY件まで）
history_container = st.container(height=500)
with history_container:
    for entry in st.session_state["chat_history"][-MAX_HISTORY_DISPLAY:]:
        role = entry["role"]
        if role == "user":
            st.chat_message("user").write(entry["content"])
        else:
            # assistant応答は「interpretation/普通テキスト」「df（データフレーム）」「chart_result（画像）」の3パターン
            with st.chat_message("assistant"):
                if "error" in entry and entry["error"]: # Simplified check
                    st.error(entry['error']) # Display user-friendly error directly

                if "interpretation" in entry:
                    st.write(entry["interpretation"])

            # --- Displaying latest_df (potentially multiple DataFrames) ---
            if "latest_df" in entry and entry["latest_df"] is not None:
                latest_df_data = entry["latest_df"]
                if isinstance(latest_df_data, dict): # New OrderedDict format
                    if not latest_df_data: # Empty dict
                        st.write("取得されたデータはありません。")
                    for req_string, df_data_list in latest_df_data.items():
                        st.write(f"データ: 「{req_string}」")
                        if df_data_list:
                            try:
                                df_disp = pd.DataFrame(df_data_list)
                                st.dataframe(df_disp)
                            except Exception as e:
                                st.error(f"DataFrame表示エラー ({req_string}): {e}")
                                st.write(df_data_list) # Display raw data if error
                        else:
                            st.write("(この要件に対するデータはありません)")
                elif isinstance(latest_df_data, list): # Fallback for old list format
                    if latest_df_data:
                        try:
                            df_disp = pd.DataFrame(latest_df_data)
                            st.dataframe(df_disp)
                        except Exception as e:
                            st.error(f"DataFrame表示エラー (旧形式): {e}")
                            st.write(latest_df_data)
                    else:
                        st.write("取得されたデータはありません。 (旧形式)")
                # else: could be other types if state is corrupted, ignore for now

            if "chart_result" in entry and entry["chart_result"]:
                chart_img = base64.b64decode(entry["chart_result"])
                st.image(chart_img, caption="AI生成グラフ")

# 3.2 ユーザー入力受付
user_input = st.chat_input(
    "質問を入力してください（例: 'カテゴリごとの合計販売金額を出して'）",
    disabled=st.session_state.get("disabled", False)
)
if user_input:
    # --- 4. ユーザー入力バブルを即時表示 & 入力フィールドを無効化 ---
    st.chat_message("user").write(user_input)
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    st.session_state.disabled = True
    st.rerun() # Immediately disable input and refresh to show user message before "thinking"

    # --- 5. AIバブル(typing演出) ---
    # This part will run after the rerun caused by disabling the input
    ai_msg_placeholder = st.empty()
    # Only show typing if it's actually processing (i.e., input was just submitted)
    # This check helps prevent re-showing typing animation on subsequent reruns if state changes elsewhere
    if st.session_state.get("disabled", False): # Check if we are in "processing" mode
        for i in range(8):  # 約4秒間タイピング演出
            dots = "." * ((i % 4) + 1)
            ai_msg_placeholder.chat_message("assistant").write(f"AI is thinking{dots} _typing_ :speech_balloon:")
            time.sleep(0.5)

    # --- 6. LangGraphバックエンド呼び出し ---
    config = {'configurable': {'thread_id': '1'}}
    # Retrieve the latest user_input from chat_history as user_input variable might be from previous run post rerun
    current_user_input = st.session_state["chat_history"][-1]["content"]
    input_state = dict(st.session_state.get("memory_state", {}))
    input_state["input"] = current_user_input # Use the most recent input
    res = compiled_workflow.invoke(input_state, config)

    # --- 7. AIバブル差し替え ---
    with ai_msg_placeholder.chat_message("assistant"): # ai_msg_placeholder should still be valid
        if "error" in res and res["error"]: # Simplified check
            st.error(res['error']) # Display user-friendly error directly

        if "interpretation" in res and res["interpretation"]: # Ensure not None or empty
            st.write(res["interpretation"])

        # --- Displaying latest_df from live response ---
        if "latest_df" in res and res["latest_df"] is not None:
            latest_df_data = res["latest_df"]
            if isinstance(latest_df_data, dict): # New OrderedDict format
                if not latest_df_data:
                     st.write("取得されたデータはありません。")
                for req_string, df_data_list in latest_df_data.items():
                    st.write(f"データ: 「{req_string}」")
                    if df_data_list:
                        try:
                            df_disp = pd.DataFrame(df_data_list)
                            st.dataframe(df_disp)
                        except Exception as e:
                            st.error(f"DataFrame表示エラー ({req_string}): {e}")
                            st.write(df_data_list)
                    else:
                        st.write("(この要件に対するデータはありません)")
            elif isinstance(latest_df_data, list): # Fallback for old list format
                if latest_df_data:
                    try:
                        df_disp = pd.DataFrame(latest_df_data)
                        st.dataframe(df_disp)
                    except Exception as e:
                        st.error(f"DataFrame表示エラー (旧形式): {e}")
                        st.write(latest_df_data)
                else:
                    st.write("取得されたデータはありません。 (旧形式)")
            # else: could be other types if state is corrupted, ignore for now

        if "chart_result" in res and res["chart_result"]:
            st.image(base64.b64decode(res["chart_result"]), caption="AI生成グラフ")

    # --- 8. 履歴に保存 & 入力フィールドを再度有効化 ---
    st.session_state["chat_history"].append({"role": "assistant", **res})
    st.session_state["memory_state"] = res # Save the full state for potential resume

    st.session_state.disabled = False
    st.rerun() # Re-enable input and refresh UI
