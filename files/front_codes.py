import streamlit as st
import base64
import pandas as pd
# from PIL import Image # Unused
# from io import BytesIO # Unused
import time
from files.backend_codes import build_workflow
import uuid # Add this import
import collections # Add this import

# compiled_workflow = build_workflow() # Redundant, get_workflow() is used

# --- 1. セッション状態で履歴＆state管理 ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "memory_state" not in st.session_state:
    st.session_state["memory_state"] = {} # This will store the full state from the backend
if "disabled" not in st.session_state:
    st.session_state["disabled"] = False
if "session_thread_id" not in st.session_state: # Ensure session_thread_id is initialized
    st.session_state["session_thread_id"] = str(uuid.uuid4())
if "awaiting_clarification_input" not in st.session_state:
    st.session_state.awaiting_clarification_input = False
if "clarification_question_text" not in st.session_state:
    st.session_state.clarification_question_text = None
if "user_input_type" not in st.session_state: # Added for multi-stage plan interaction
    st.session_state["user_input_type"] = None

# --- 2. ワークフローのセットアップ（必要ならキャッシュ） ---
MAX_HISTORY_DISPLAY = 20 # Max number of chat messages to display

@st.cache_resource
def get_workflow():
    return build_workflow()

compiled_workflow = get_workflow()

# --- Clear History Button ---
if st.button("チャット履歴をクリア"):
    st.session_state.chat_history = []
    st.session_state.memory_state = {} # Clear local frontend state
    st.session_state.awaiting_clarification_input = False # Reset clarification state
    st.session_state.clarification_question_text = None # Reset clarification state
    if "user_selected_option" in st.session_state: # Though not used in current flow, good practice if added
        del st.session_state.user_selected_option
    if "analysis_options" in st.session_state.get("memory_state", {}): # Clear from memory_state
        del st.session_state.memory_state["analysis_options"]


    # Use the session-specific thread_id for clearing (already initialized)
    session_specific_thread_id = st.session_state["session_thread_id"]
    config = {'configurable': {'thread_id': session_specific_thread_id}}
    try:
        # Directly invoke the backend with the special clear command
        compiled_workflow.invoke({"input": "SYSTEM_CLEAR_HISTORY"}, config=config)
        st.success(f"セッションのチャット履歴と関連するバックエンドの状態がクリアされました。")
    except Exception as e:
        st.error(f"セッション {session_specific_thread_id} のバックエンド状態のクリア中にエラーが発生しました: {e}")
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
                # Display multi-stage analysis progress if applicable
                if "analysis_plan" in entry and entry["analysis_plan"] and entry.get("current_plan_step_index") is not None:
                    plan = entry["analysis_plan"]
                    current_idx = entry["current_plan_step_index"]
                    total_steps = len(plan)
                    if 0 <= current_idx < total_steps: # Ensure index is valid
                        current_step_info = plan[current_idx]
                        action = current_step_info.get('action', 'N/A')
                        details = current_step_info.get('details', 'N/A')
                        st.info(f"複数ステップ分析: ステップ {current_idx + 1} / {total_steps} (アクション: {action}, 詳細: {details})")

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

# 3.2 ユーザー入力受付 & Clarification Handling
if st.session_state.awaiting_clarification_input and st.session_state.clarification_question_text:
    # --- Display Clarification Question and Get Answer ---
    st.info(st.session_state.clarification_question_text) # Display the question

    with st.form(key="clarification_form"):
        clarification_answer = st.text_input("要求された情報を入力してください:")
        submit_clarification_button = st.form_submit_button(label="明確化情報を送信")

    if submit_clarification_button and clarification_answer:
        st.session_state["chat_history"].append({"role": "user", "content": clarification_answer, "type": "clarification_answer"})

        # current_memory_state = dict(st.session_state.get("memory_state", {})) # Unused assignment
        # current_memory_state["user_clarification"] = clarification_answer # This was not updating session_state.memory_state
        # The user_clarification is correctly added to invoke_payload later.

        st.session_state.disabled = True # Disable main input during processing
        st.session_state.awaiting_clarification_input = False
        st.session_state.clarification_question_text = None
        st.rerun() # Rerun to show user's clarification and start AI "thinking"

# Main chat input - disabled if awaiting clarification OR if processing normal input OR if awaiting step confirmation
chat_input_disabled = st.session_state.disabled or \
                      st.session_state.awaiting_clarification_input or \
                      st.session_state.get("memory_state", {}).get("awaiting_step_confirmation", False)

user_input = st.chat_input(
    "質問を入力してください（例: 'カテゴリごとの合計販売金額を出して'）",
    disabled=chat_input_disabled
)

# --- Multi-stage Analysis Controls (Proceed/Cancel) ---
if st.session_state.get("memory_state", {}).get("awaiting_step_confirmation"):
    # Display intermediate results (already handled by chat history loop)
    st.markdown("---") # Separator
    st.write("分析は一時停止中です。アクションを選択してください:")
    button_cols = st.columns([1, 1, 2]) # Adjust column ratios as needed

    if button_cols[0].button("次のステップへ進む", key="proceed_step_button"):
        st.session_state["user_input_type"] = "proceed_analysis_step"
        # The memory_state (which includes the plan) is already set from the last backend response.
        # We just need to signal the intent to proceed.
        st.session_state.disabled = True # Trigger backend processing
        st.rerun()

    if button_cols[1].button("分析計画をキャンセル", key="cancel_plan_button"):
        st.session_state["user_input_type"] = "cancel_analysis_plan"
        st.session_state.disabled = True # Trigger backend processing
        st.rerun()

if user_input and not st.session_state.awaiting_clarification_input and not st.session_state.get("memory_state", {}).get("awaiting_step_confirmation", False): # Normal user input processing
    # --- 4. ユーザー入力バブルを即時表示 & 入力フィールドを無効化 ---
    st.chat_message("user").write(user_input)
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state.disabled = True
    st.rerun() # Immediately disable input and refresh to show user message

# This block handles processing after either a new message or a clarification is submitted
# It relies on st.session_state.disabled being True if processing is needed.
if st.session_state.disabled: # This will be true after user submits new input OR clarification (due to rerun)
    # --- 5. AIバブル(typing演出) ---
    ai_msg_placeholder = st.empty()
    for i in range(8):  # 約4秒間タイピング演出
        dots = "." * ((i % 4) + 1)
        ai_msg_placeholder.chat_message("assistant").write(f"AIが思考中{dots} _入力中_ :speech_balloon:")
        time.sleep(0.5)

    # --- 6. LangGraphバックエンド呼び出し ---
    # Use the session-specific thread_id
    config = {'configurable': {'thread_id': st.session_state["session_thread_id"]}}
    # current_user_input = st.session_state["chat_history"][-1]["content"] # Unused variable

    user_action_type = st.session_state.get("user_input_type")
    current_memory = dict(st.session_state.get("memory_state", {}))
    invoke_payload = {}

    if user_action_type == "proceed_analysis_step":
        # Backend needs the current memory state to know about the plan, index etc.
        # And user_action to be routed correctly.
        invoke_payload = {**current_memory, "user_action": "proceed_analysis_step", "awaiting_step_confirmation": False}
        st.session_state["chat_history"].append({"role": "user", "content": "ユーザーは次のステップへ進むことを選択しました。", "type": "system_action"})
    elif user_action_type == "cancel_analysis_plan":
        invoke_payload = {**current_memory, "user_action": "cancel_analysis_plan", "awaiting_step_confirmation": False}
        st.session_state["chat_history"].append({"role": "user", "content": "ユーザーは分析計画のキャンセルを選択しました。", "type": "system_action"})
    else:
        # This is the existing logic for handling normal user input, clarifications, or analysis option selections.
        last_user_message_entry = next((msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
        if last_user_message_entry:
            last_user_interaction_type = last_user_message_entry.get("type")
            current_user_input_content = last_user_message_entry["content"]

            if last_user_interaction_type == "clarification_answer":
                invoke_payload = {**current_memory, "user_clarification": current_user_input_content}
            elif last_user_interaction_type == "analysis_selection":
                # This logic was for when analysis options were selected.
                # It constructed a new_input_context. We should ensure it's compatible.
                # The key is that 'input' becomes the selected option, and history is preserved.
                chat_history_for_query_context = st.session_state.get("chat_history", [])
                query_history_list = [msg["content"] for msg in chat_history_for_query_context[:-1] if msg["role"] == "user"]
                invoke_payload = {
                    "input": current_user_input_content, # Selected option
                    "df_history": current_memory.get("df_history", []),
                    "query_history": query_history_list,
                    "latest_df": collections.OrderedDict(),
                    "SQL": None, "interpretation": None, "chart_result": None,
                    "clarification_question": None, "user_clarification": None,
                    "analysis_options": None, "error": None, "intent_list": [],
                    "data_requirements": [], "missing_data_requirements": None,
                    # Carry over plan variables if any, though analysis_selection might imply starting fresh
                    # or modifying an existing plan - for now, this assumes it's a new primary query.
                    "analysis_plan": current_memory.get("analysis_plan"),
                    "current_plan_step_index": current_memory.get("current_plan_step_index"),
                    "awaiting_step_confirmation": current_memory.get("awaiting_step_confirmation"),
                    "complex_analysis_original_query": current_memory.get("complex_analysis_original_query"),
                }
            else: # Standard new query from the main chat_input
                previous_df_history = current_memory.get("df_history", [])
                full_chat_history_for_query_context = st.session_state.get("chat_history", [])
                query_history_context = [msg["content"] for msg in full_chat_history_for_query_context[:-1] if msg["role"] == "user"]
                invoke_payload = {
                    "input": current_user_input_content,
                    "df_history": previous_df_history,
                    "query_history": query_history_context,
                    "latest_df": collections.OrderedDict(), "SQL": None, "interpretation": None, "chart_result": None,
                    "clarification_question": None, "user_clarification": None, "analysis_options": None,
                    "error": None, "intent_list": [], "data_requirements": [], "missing_data_requirements": None,
                    # Ensure plan variables are reset/None for a brand new query if not part of a plan context
                    "analysis_plan": None, "current_plan_step_index": None,
                    "awaiting_step_confirmation": False, "complex_analysis_original_query": None,
                }
        else: # Should not happen if st.session_state.disabled is True due to user input
            st.session_state.disabled = False
            st.rerun()
            return

    # Clear the user_input_type after processing
    st.session_state["user_input_type"] = None

    if not invoke_payload:
        st.warning("問題が発生しました。実行するアクションがありません。もう一度お試しください。")
        st.session_state.disabled = False
        st.rerun()
        return

    res = compiled_workflow.invoke(invoke_payload, config=config)

    # --- 7. AIバブル差し替え & Clarification Check ---
    if "clarification_question" in res and res["clarification_question"]:
            st.session_state.awaiting_clarification_input = True
            st.session_state.clarification_question_text = res["clarification_question"]
            st.session_state["chat_history"].append({"role": "assistant", "content": res["clarification_question"], "type": "clarification_request"})
            st.session_state["memory_state"] = res
            ai_msg_placeholder.empty()
            st.session_state.disabled = False
            st.rerun()
        else:
            # Normal response processing
            with ai_msg_placeholder.chat_message("assistant"):
                if "error" in res and res["error"]:
                    st.error(res['error'])

                if "interpretation" in res and res["interpretation"]:
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

        # --- Display Analysis Options (if any) ---
        if "analysis_options" in res and res["analysis_options"] and isinstance(res["analysis_options"], list) and len(res["analysis_options"]) > 0:
            st.markdown("---") # Separator
            st.write("提案される次のステップ:")
            for i, option_text in enumerate(res["analysis_options"]):
                if not isinstance(option_text, str): # Ensure option_text is a string
                    continue # Skip if not a string
                button_key = f"analysis_option_btn_{uuid.uuid4().hex}_{i}" # Unique key
                if st.button(option_text, key=button_key):
                    # User clicked an analysis option
                    st.chat_message("user").write(option_text)
                    st.session_state["chat_history"].append({"role": "user", "content": option_text, "type": "analysis_selection"})

                    current_memory_state = st.session_state.get("memory_state", {})
                    # Prepare a clean state for the new turn, but carry over history

                    # query_history for backend should be a list of user query strings
                    # up to, but not including, the current selected option.
                    # The current selected option becomes the new "input".
                    chat_history_for_query_context = st.session_state.get("chat_history", [])
                    # The last item in chat_history_for_query_context is the selected option itself (just added),
                    # so we take up to [:-1] for the history that *led* to this option.
                    query_history_list = [msg["content"] for msg in chat_history_for_query_context[:-1] if msg["role"] == "user"]

                    new_input_context = {
                        "input": option_text,
                        "df_history": current_memory_state.get("df_history", []),
                        "query_history": query_history_list,
                        # Reset other fields for a fresh analysis based on the selected option
                        "latest_df": collections.OrderedDict(),
                        "SQL": None,
                        "interpretation": None,
                        "chart_result": None,
                        "clarification_question": None,
                        "user_clarification": None,
                        "analysis_options": None, # Clear options for the new turn
                        "error": None,
                        "intent_list": [], # Will be re-classified
                        "data_requirements": [], # Will be re-extracted
                        "missing_data_requirements": None,
                    }
                    st.session_state.memory_state = new_input_context
                    st.session_state.disabled = True # Trigger backend processing
                    # No need to call ai_msg_placeholder.empty() here as it's outside this with block
                    st.rerun()
                    break # Exit loop once a button is clicked and rerun is triggered

    # --- 8. 履歴に保存 & 入力フィールドを再度有効化 ---
    # Ensure this runs only if we haven't already rerun due to button click
    if not (st.session_state.disabled and any(entry.get("type") == "analysis_selection" for entry in st.session_state.chat_history[-1:])):
        st.session_state["chat_history"].append({"role": "assistant", **res})
        st.session_state["memory_state"] = res # Save the full state

        st.session_state.disabled = False
        st.rerun() # Re-enable input and refresh UI
