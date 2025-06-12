import streamlit as st
import pandas as pd
import json
import uuid
from backend_codes import build_workflow
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import logging
import plotly.io as pio
import plotly.graph_objects as go
from functions import render_plan_sidebar, extract_alerts

logger = logging.getLogger("langgraph")
logger.setLevel(logging.DEBUG)

# --- 1. ワークフローの準備（初回実行時のみ） ---
# @st.cache_resourceを使うことで、アプリの再実行時にワークフローを再構築するのを防ぎ、高速化します。
@st.cache_resource
def get_workflow():
    return build_workflow()

compiled_workflow = get_workflow()

# --- 2. Streamlitページの基本設定 ---
st.title("データ分析チャットAI")
st.caption("このAIは、裏側でLangGraphというフレームワークで動作しています。")

# -セッションごとのチャット履歴を初期化 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# セッションIDを自前で管理する場合
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
# AI から追加情報を求められているか
if "awaiting_ai_question" not in st.session_state:
    st.session_state["awaiting_ai_question"] = False   
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = ""
#分析plan進捗管理用のstate
if "plan_steps" not in st.session_state:
    st.session_state["plan_steps"] = []
if "plan_cursor" not in st.session_state:
    st.session_state["plan_cursor"] = -1
#error表示用
if "error_log" not in st.session_state:
    st.session_state["error_log"] = []

# --- 4. 過去のメッセージをすべて表示 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # contentが文字列の場合（通常の会話）
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        # contentが辞書の場合（データやグラフを含む）
        elif isinstance(message["content"], dict):
            if "result_df_json" in message["content"] and message["content"]["result_df_json"]:
                df_data = json.loads(message["content"]["result_df_json"])
                st.dataframe(pd.DataFrame(df_data))
            if "fig_json" in message["content"] and message["content"]["fig_json"]:
                fig_dict = pio.from_json(message["content"]["fig_json"], output_type="dict")
                fig = go.Figure(fig_dict)
                st.plotly_chart(fig, use_container_width=True)
            if "interpretation_text" in message["content"] and message["content"]["interpretation_text"]:
                st.markdown(message["content"]["interpretation_text"])


# ユーザーからの入力を受け付け
if st.session_state.awaiting_ai_question:
    prompt_label = "AI からの質問に回答してください" 
else:
    prompt_label = "分析したいことを入力してください（例: カテゴリ別の売上を見せて）"
user_input = st.chat_input(prompt_label)


if user_input:
    # ユーザーの入力を履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
     # もしユーザー質問フェーズなら、userinputがあった時点で解除
    if st.session_state.awaiting_ai_question:
        st.session_state.awaiting_ai_question = False
        st.session_state.pending_question = ""

    with st.chat_message("user"):
        st.markdown(user_input)

    # AIの応答を待つ間、スピナー（くるくる回るアイコン）を表示
    # アシスタントの応答スペースを確保
    with st.chat_message("assistant"):
        
        # st.spinnerの代わりにst.statusを使用。実行状況を動的に更新できる
        with st.status("AIがリクエストを分析中...", expanded=True) as status:
            try:
                # LangGraphバックエンドの呼び出し設定
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                input_data = {"messages": [HumanMessage(content=user_input)]}
                
                # .invoke()の代わりに.stream()を使い、リアルタイムでチャンクを処理
                for chunk in compiled_workflow.stream(input_data, config, stream_mode="updates"):
                    
                    #ユーザーへの質問モードであれば待機フラグを立てる
                    if "ask_user_node" in chunk:
                        st.session_state.awaiting_ai_question = True
                        #supervisorが質問をしていたらその質問内容を取得
                        if "supervisor" in chunk:
                            sup_msgs = chunk["supervisor"]["messages"]
                            if sup_msgs and isinstance(sup_msgs[0], AIMessage):
                                st.session_state.pending_question = sup_msgs[0].content
                    
                    # chunkにplanが出現したらstateに格納    
                    if "plan" in chunk:
                        st.session_state.plan_steps = chunk["plan"]
                    if "plan_cursor" in chunk:
                        st.session_state.plan_cursor = chunk["plan_cursor"] 
                
                    # --- chunkを解析して、実行状況をリアルタイムで表示 ---
                    if "supervisor" in chunk:
                        # スーパーバイザーが思考中
                        status.update(label=f"どの専門家に依頼するか思考中...")
                    elif "sql_node" in chunk:
                        # SQLノードが実行中
                        status.update(label="データベースにアクセスし、SQLを実行中...")
                    elif "processing_node" in chunk:
                        # データ加工/グラフ作成ノードが実行中
                        status.update(label="データを加工し、グラフを作成中...")
                    elif "interpret_node" in chunk:
                        # 解釈ノードが実行中
                        status.update(label="結果を解釈し、説明を生成中...")
                    elif "metadata_retrieval_node" in chunk:
                        # テーブル情報ノードが実行中
                        status.update(label="テーブル情報を取得中...")
                    else:
                        status.update(label="少々お待ちください...")
                    
                    #エラーがあった場合はsession_stateに追加
                    alerts = extract_alerts(chunk)
                    if alerts:
                        st.session_state.error_log.extend(alerts)

                    #エラーログはsidebarに表示
                    if len(st.session_state.error_log)>0:
                         with st.sidebar.expander("エラーログ", expanded=False):
                             for rec in reversed(st.session_state.error_log[-50:]):  # 直近50件
                                 status = rec["status"]
                                 node = rec["node"]
                                 summary = rec["summary"]
                                 st.markdown(f"{node}_{status}:{summary}")

                status.update(label="分析完了！", state="complete", expanded=False)
                ai_response = chunk["supervisor"]["messages"][0]
                response_content = None

                # ユーザーへの質問モードであれば、質問内容を返答内容とする。
                if st.session_state.awaiting_ai_question:
                     response_content = st.session_state.pending_question
                else:
                    if isinstance(ai_response, AIMessage):
                        # 質問モードでなく、かつ、返答がAIMessageであれば通常のテキスト応答
                        response_content = ai_response.content
                    else:
                        response_content = "AI応答の最終状態が取得できませんでした。"

                # 解析したAIの応答を履歴に追加
                st.session_state.messages.append({"role": "assistant", "content": response_content})             

                # ToolMessageの場合、result_payload(figやdf,interpretなど)の内容を追加
                if isinstance(ai_response, ToolMessage):
                    payload = json.loads(ai_response.content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": payload["result_payload"]
                    })

                # plan進捗を表示
                render_plan_sidebar()
                st.rerun()
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
