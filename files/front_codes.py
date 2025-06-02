import streamlit as st
import base64
import pandas as pd
# from PIL import Image # 未使用
# from io import BytesIO # 未使用
# import time # Streamlitのst.statusを使用するため、手動のtime.sleepは不要になります
from files.backend_codes import build_workflow
import uuid # このインポートを追加
import collections # このインポートを追加

# compiled_workflow = build_workflow() # 冗長、get_workflow()が使用されます

# --- 1. セッション状態で履歴＆state管理 ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "memory_state" not in st.session_state:
    st.session_state["memory_state"] = {} # これにはバックエンドからの完全な状態が保存されます
if "disabled" not in st.session_state:
    st.session_state["disabled"] = False
if "session_thread_id" not in st.session_state: # session_thread_idが初期化されていることを確認してください
    st.session_state["session_thread_id"] = str(uuid.uuid4())
if "awaiting_clarification_input" not in st.session_state:
    st.session_state.awaiting_clarification_input = False
if "clarification_question_text" not in st.session_state:
    st.session_state.clarification_question_text = None
if "user_input_type" not in st.session_state: # マルチステップ計画の対話のために追加
    st.session_state["user_input_type"] = None

# --- 2. ワークフローのセットアップ（必要ならキャッシュ） ---
MAX_HISTORY_DISPLAY = 20 # 表示するチャットメッセージの最大数

@st.cache_resource
def get_workflow():
    return build_workflow()

compiled_workflow = get_workflow()

# --- チャット履歴クリアボタン ---
if st.button("チャット履歴をクリア"):
    st.session_state.chat_history = []
    st.session_state.memory_state = {} # ローカルフロントエンドの状態をクリア
    st.session_state.awaiting_clarification_input = False # 明確化の状態をリセット
    st.session_state.clarification_question_text = None # 明確化の状態をリセット
    if "user_selected_option" in st.session_state: # 現在のフローでは使用されていませんが、追加する場合のベストプラクティス
        del st.session_state.user_selected_option
    if "analysis_options" in st.session_state.get("memory_state", {}): # memory_stateからクリア
        del st.session_state.memory_state["analysis_options"]


    # クリアにはセッション固有のthread_idを使用します（初期化済み）
    session_specific_thread_id = st.session_state["session_thread_id"]
    config = {'configurable': {'thread_id': session_specific_thread_id}}
    try:
        # 特別なクリアコマンドでバックエンドを直接呼び出します
        compiled_workflow.invoke({"input": "SYSTEM_CLEAR_HISTORY"}, config=config)
        st.success(f"セッションのチャット履歴と関連するバックエンドの状態がクリアされました。")
    except Exception as e:
        st.error(f"セッション {session_specific_thread_id} のバックエンド状態のクリア中にエラーが発生しました: {e}")
    st.rerun() # クリア後にUIをクリーンにリフレッシュするために再実行

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
                # 該当する場合、マルチステップ分析の進捗を表示
                if "analysis_plan" in entry and entry["analysis_plan"] and entry.get("current_plan_step_index") is not None:
                    plan = entry["analysis_plan"]
                    current_idx = entry["current_plan_step_index"]
                    total_steps = len(plan)
                    if 0 <= current_idx < total_steps: # インデックスが有効であることを確認
                        current_step_info = plan[current_idx]
                        action = current_step_info.get('action', 'N/A')
                        details = current_step_info.get('details', 'N/A')
                        st.info(f"複数ステップ分析: ステップ {current_idx + 1}/{total_steps} (アクション: {action}, 詳細: {details})")

                if "error" in entry and entry["error"]: # 簡略化されたチェック
                    st.error(entry['error']) # ユーザーフレンドリーなエラーを直接表示

                if "interpretation" in entry:
                    st.write(entry["interpretation"])

            # --- latest_dfの表示（複数のDataFrameの可能性あり）---
            if "latest_df" in entry and entry["latest_df"] is not None:
                latest_df_data = entry["latest_df"]
                if isinstance(latest_df_data, dict): # 新しいOrderedDict形式
                    if not latest_df_data: # 空の辞書
                        st.write("取得されたデータはありません。")
                    for req_string, df_data_list in latest_df_data.items():
                        st.write(f"データ: 「{req_string}」")
                        if df_data_list:
                            try:
                                df_disp = pd.DataFrame(df_data_list)
                                st.dataframe(df_disp)
                            except Exception as e:
                                st.error(f"DataFrame表示エラー ({req_string}): {e}")
                                st.write(df_data_list) # エラーの場合、生データを表示
                        else:
                            st.write("(この要件に対するデータはありません)")
                elif isinstance(latest_df_data, list): # 古いリスト形式のフォールバック
                    if latest_df_data:
                        try:
                            df_disp = pd.DataFrame(latest_df_data)
                            st.dataframe(df_disp)
                        except Exception as e:
                            st.error(f"DataFrame表示エラー (旧形式): {e}")
                            st.write(latest_df_data)
                    else:
                        st.write("取得されたデータはありません。 (旧形式)")
                # else: 状態が破損している場合、他の型である可能性があります。とりあえず無視します

            if "chart_result" in entry and entry["chart_result"]:
                chart_img = base64.b64decode(entry["chart_result"])
                st.image(chart_img, caption="AI生成グラフ")

# 3.2 ユーザー入力受付 & Clarification Handling
if st.session_state.awaiting_clarification_input and st.session_state.clarification_question_text:
    # --- 明確化の質問を表示して回答を取得 ---
    st.info(st.session_state.clarification_question_text) # 質問を表示

    with st.form(key="clarification_form"):
        clarification_answer = st.text_input("要求された情報を提供してください:")
        submit_clarification_button = st.form_submit_button(label="明確化を送信")

    if submit_clarification_button and clarification_answer:
        st.session_state["chat_history"].append({"role": "user", "content": clarification_answer, "type": "clarification_answer"})

        # current_memory_state = dict(st.session_state.get("memory_state", {})) # 未使用の代入
        # current_memory_state["user_clarification"] = clarification_answer # これはsession_state.memory_stateを更新していませんでした
        # user_clarificationは後で正しくinvoke_payloadに追加されます。

        st.session_state.disabled = True # 処理中はメイン入力を無効化
        st.session_state.awaiting_clarification_input = False
        st.session_state.clarification_question_text = None
        st.rerun() # ユーザーの明確化を表示し、AIの「思考」を開始するために再実行

# メインチャット入力 - 明確化待機中、通常入力処理中、またはステップ確認待機中の場合は無効
chat_input_disabled = st.session_state.disabled or \
                      st.session_state.awaiting_clarification_input or \
                      st.session_state.get("memory_state", {}).get("awaiting_step_confirmation", False)

user_input = st.chat_input(
    "質問を入力してください（例: 'カテゴリごとの合計販売金額を出して'）",
    disabled=chat_input_disabled
)

# --- マルチステップ分析コントロール（続行/キャンセル）---
if st.session_state.get("memory_state", {}).get("awaiting_step_confirmation"):
    # 中間結果を表示（チャット履歴ループで処理済み）
    st.markdown("---") # 区切り線
    st.write("分析は一時停止中です。アクションを選択してください:")
    button_cols = st.columns([1, 1, 2]) # 必要に応じて列の比率を調整

    if button_cols[0].button("次のステップに進む", key="proceed_step_button"):
        st.session_state["user_input_type"] = "proceed_analysis_step"
        # memory_state（プランを含む）は、最後のバックエンド応答から既に設定されています。
        # 続行する意図を通知するだけで済みます。
        st.session_state.disabled = True # バックエンド処理をトリガー
        st.rerun()

    if button_cols[1].button("分析計画をキャンセル", key="cancel_plan_button"):
        st.session_state["user_input_type"] = "cancel_analysis_plan"
        st.session_state.disabled = True # バックエンド処理をトリガー
        st.rerun()

if user_input and not st.session_state.awaiting_clarification_input and not st.session_state.get("memory_state", {}).get("awaiting_step_confirmation", False): # 通常のユーザー入力処理
    # --- 4. ユーザー入力バブルを即時表示 & 入力フィールドを無効化 ---
    st.chat_message("user").write(user_input)
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state.disabled = True
    st.rerun() # 入力を直ちに無効にし、ユーザーメッセージを表示するためにリフレッシュ

# このブロックは、新しいメッセージまたは明確化が送信された後の処理を扱います
# 処理が必要な場合、st.session_state.disabledがTrueであることに依存します。
if st.session_state.disabled: # これは、ユーザーが新しい入力または明確化を送信した後（再実行のため）にtrueになります
    # --- 5. AIバブル(typing演出) ---
    ai_msg_placeholder = st.empty() # 以前のメッセージをクリアするために保持、またはst.statusが完全に置き換える場合は削除検討
    # for i in range(8):  # 約4秒間タイピング演出
    #     dots = "." * ((i % 4) + 1)
    #     ai_msg_placeholder.chat_message("assistant").write(f"AIが思考中です{dots} _タイピング中_ :speech_balloon:")
    #     time.sleep(0.5)
    ai_msg_placeholder.empty() # st.statusを使う前にプレースホルダーをクリア

    # --- 6. LangGraphバックエンド呼び出し ---
    # セッション固有のthread_idを使用
    config = {'configurable': {'thread_id': st.session_state["session_thread_id"]}}
    # current_user_input = st.session_state["chat_history"][-1]["content"] # 未使用の変数

    user_action_type = st.session_state.get("user_input_type")
    current_memory = dict(st.session_state.get("memory_state", {}))
    invoke_payload = {}

    if user_action_type == "proceed_analysis_step":
        # バックエンドは、プランやインデックスなどを知るために現在のメモリ状態を必要とします。
        # そして、user_actionが正しくルーティングされるように。
        invoke_payload = {**current_memory, "user_action": "proceed_analysis_step", "awaiting_step_confirmation": False}
        st.session_state["chat_history"].append({"role": "user", "content": "ユーザーは次のステップに進むことを選択しました。", "type": "system_action"})
    elif user_action_type == "cancel_analysis_plan":
        invoke_payload = {**current_memory, "user_action": "cancel_analysis_plan", "awaiting_step_confirmation": False}
        st.session_state["chat_history"].append({"role": "user", "content": "ユーザーは分析計画をキャンセルすることを選択しました。", "type": "system_action"})
    else:
        # これは、通常のユーザー入力、明確化、または分析オプションの選択を処理するための既存のロジックです。
        last_user_message_entry = next((msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
        if last_user_message_entry:
            last_user_interaction_type = last_user_message_entry.get("type")
            current_user_input_content = last_user_message_entry["content"]

            if last_user_interaction_type == "clarification_answer":
                invoke_payload = {**current_memory, "user_clarification": current_user_input_content}
            elif last_user_interaction_type == "analysis_selection":
                # このロジックは、分析オプションが選択された場合のためのものでした。
                # new_input_contextを構築しました。互換性があることを確認する必要があります。
                #重要なのは、「input」が選択されたオプションになり、履歴が保持されることです。
                chat_history_for_query_context = st.session_state.get("chat_history", [])
                query_history_list = [msg["content"] for msg in chat_history_for_query_context[:-1] if msg["role"] == "user"]
                invoke_payload = {
                    "input": current_user_input_content, # 選択されたオプション
                    "df_history": current_memory.get("df_history", []),
                    "query_history": query_history_list,
                    "latest_df": collections.OrderedDict(),
                    "SQL": None, "interpretation": None, "chart_result": None,
                    "clarification_question": None, "user_clarification": None,
                    "analysis_options": None, "error": None, "intent_list": [],
                    "data_requirements": [], "missing_data_requirements": None,
                    # プラン変数があれば引き継ぎます（分析の選択は新規開始を意味するかもしれませんが）
                    # または既存のプランを変更します - 現時点では、これは新しいプライマリクエリであると想定しています。
                    "analysis_plan": current_memory.get("analysis_plan"),
                    "current_plan_step_index": current_memory.get("current_plan_step_index"),
                    "awaiting_step_confirmation": current_memory.get("awaiting_step_confirmation"),
                    "complex_analysis_original_query": current_memory.get("complex_analysis_original_query"),
                }
            else: # メインchat_inputからの標準的な新規クエリ
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
                    # プランコンテキストの一部でない場合、全く新しいクエリに対してプラン変数がリセット/Noneであることを確認
                    "analysis_plan": None, "current_plan_step_index": None,
                    "awaiting_step_confirmation": False, "complex_analysis_original_query": None,
                }
        else: # ユーザー入力によりst.session_state.disabledがTrueの場合は発生しません
            st.session_state.disabled = False
            st.rerun()
            return

    # 処理後にuser_input_typeをクリア
    st.session_state["user_input_type"] = None

    if not invoke_payload:
        st.warning("問題が発生しました。実行できるアクションがありません。もう一度お試しください。")
        st.session_state.disabled = False
        st.rerun()
        return

    # res = compiled_workflow.invoke(invoke_payload, config=config) # invokeからstreamに変更
    stream = compiled_workflow.stream(invoke_payload, config=config)
    
    final_res = None
    with st.status("AIが処理を実行中...", expanded=True) as status_ui:
        for event in stream:
            current_state_snapshot = None
            if event:
                event_key = list(event.keys())[0] 
                if event_key and isinstance(event[event_key], dict):
                     current_state_snapshot = event[event_key]

            if current_state_snapshot and "current_status_message" in current_state_snapshot and current_state_snapshot["current_status_message"]:
                status_ui.update(label=current_state_snapshot["current_status_message"])
            
            if current_state_snapshot: 
                final_res = current_state_snapshot
    
    res = final_res
    if res is None: 
        res = {} 
        st.warning("AIからの応答が期待通りに受信できませんでした。")

    # --- 7. AIバブル差し替え & Clarification Check ---
    # ai_msg_placeholder は st.status で置き換えられたため、直接操作しない。
    # 結果の表示は st.status ブロックの外で行う。
    if "clarification_question" in res and res["clarification_question"]:
            st.session_state.awaiting_clarification_input = True
            st.session_state.clarification_question_text = res["clarification_question"]
            st.session_state["chat_history"].append({"role": "assistant", "content": res["clarification_question"], "type": "clarification_request"})
            st.session_state["memory_state"] = res
            # ai_msg_placeholder.empty() # st.statusがUIを管理
            st.session_state.disabled = False
            st.rerun()
    else:
        # 通常の応答処理をチャットメッセージとして表示
        with st.chat_message("assistant"): # このブロックは `st.status` の後、最終結果を表示するために必要
            if "error" in res and res["error"]:
                st.error(res['error'])

            if "interpretation" in res and res["interpretation"]:
                st.write(res["interpretation"])

            # --- ライブ応答からのlatest_dfの表示 ---
            if "latest_df" in res and res["latest_df"] is not None:
                latest_df_data = res["latest_df"]
                if isinstance(latest_df_data, dict): # 新しいOrderedDict形式
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
            elif isinstance(latest_df_data, list): # 古いリスト形式のフォールバック
                if latest_df_data:
                    try:
                        df_disp = pd.DataFrame(latest_df_data)
                        st.dataframe(df_disp)
                    except Exception as e:
                        st.error(f"DataFrame表示エラー (旧形式): {e}")
                        st.write(latest_df_data)
                else:
                    st.write("取得されたデータはありません。 (旧形式)")
            # else: 状態が破損している場合、他の型である可能性があります。とりあえず無視します

        if "chart_result" in res and res["chart_result"]:
            st.image(base64.b64decode(res["chart_result"]), caption="AI生成グラフ")

        # --- 分析オプションの表示（もしあれば）---
        if "analysis_options" in res and res["analysis_options"] and isinstance(res["analysis_options"], list) and len(res["analysis_options"]) > 0:
            st.markdown("---") # 区切り線
            st.write("推奨される次のステップ:")
            for i, option_text in enumerate(res["analysis_options"]):
                if not isinstance(option_text, str): # option_textが文字列であることを確認
                    continue # 文字列でない場合はスキップ
                button_key = f"analysis_option_btn_{uuid.uuid4().hex}_{i}" # 一意のキー
                if st.button(option_text, key=button_key):
                    # ユーザーが分析オプションをクリックしました
                    st.chat_message("user").write(option_text)
                    st.session_state["chat_history"].append({"role": "user", "content": option_text, "type": "analysis_selection"})

                    current_memory_state = st.session_state.get("memory_state", {})
                    # 新しいターンのためにクリーンな状態を準備しますが、履歴は引き継ぎます

                    # バックエンド用のquery_historyは、ユーザーのクエリ文字列のリストである必要があります
                    # 現在選択されているオプションまで（ただし、そのオプションは含まない）。
                    # 現在選択されているオプションが新しい「input」になります。
                    chat_history_for_query_context = st.session_state.get("chat_history", [])
                    # chat_history_for_query_contextの最後の項目は、選択されたオプション自体です（追加されたばかり）。
                    # したがって、このオプションに *至った* 履歴については[:-1]まで取得します。
                    query_history_list = [msg["content"] for msg in chat_history_for_query_context[:-1] if msg["role"] == "user"]

                    new_input_context = {
                        "input": option_text,
                        "df_history": current_memory_state.get("df_history", []),
                        "query_history": query_history_list,
                        # 選択されたオプションに基づいて新しい分析を行うために他のフィールドをリセット
                        "latest_df": collections.OrderedDict(),
                        "SQL": None,
                        "interpretation": None,
                        "chart_result": None,
                        "clarification_question": None,
                        "user_clarification": None,
                        "analysis_options": None, # 新しいターンのためにオプションをクリア
                        "error": None,
                        "intent_list": [], # 再分類されます
                        "data_requirements": [], # 再抽出されます
                        "missing_data_requirements": None,
                    }
                    st.session_state.memory_state = new_input_context
                    st.session_state.disabled = True # バックエンド処理をトリガー
                    # st.status がUIを管理するため、ai_msg_placeholder の直接操作は不要
                    st.rerun()
                    break # ボタンがクリックされて再実行がトリガーされたらループを終了

    # --- 8. 履歴に保存 & 入力フィールドを再度有効化 ---
    # ボタンクリックによる再実行がまだ行われていない場合にのみ、これを実行することを確認
    # また、resがNoneでないことも確認 (ストリームが空だった場合など)
    if res and not (st.session_state.disabled and any(entry.get("type") == "analysis_selection" for entry in st.session_state.chat_history[-1:])):
        st.session_state["chat_history"].append({"role": "assistant", **res})
        st.session_state["memory_state"] = res # 完全な状態を保存

        st.session_state.disabled = False
        st.rerun() # 入力を再度有効にし、UIをリフレッシュ
