import streamlit as st
import pandas as pd
import json
import uuid
from backend_codes import build_workflow
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. ワークフローの準備（初回実行時のみ） ---
# @st.cache_resourceを使うことで、アプリの再実行時にワークフローを再構築するのを防ぎ、高速化します。
@st.cache_resource
def get_workflow():
    return build_workflow()

compiled_workflow = get_workflow()

# --- 2. Streamlitページの基本設定 ---
st.title("🤖 データ分析チャットAI")
st.caption("このAIは、裏側でLangGraphというフレームワークで動作しています。")

# -セッションごとのチャット履歴を初期化 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# セッションIDを自前で管理する場合
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# --- 4. 過去のメッセージをすべて表示 ---
# st.session_state.messagesには、{"role": "user", "content": ...} や {"role": "assistant", "content": ...} の形式で保存
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
                fig = pd.io.json.read_json(message["content"]["fig_json"], typ='frame')
                st.plotly_chart(fig)
            if "interpretation" in message["content"] and message["content"]["interpretation"]:
                 st.markdown(message["content"]["interpretation"])


# --- 5. ユーザーからの入力を受け付け ---
# このブロックをまるごと置き換えます
# -------------------------------------------------------------------------

user_input = st.chat_input("分析したいことを入力してください（例: カテゴリ別の売上を見せて）")

if user_input:
    # ユーザーの入力を履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # アシスタントの応答スペースを確保
    with st.chat_message("assistant"):

        # st.spinnerの代わりにst.statusを使用。実行状況を動的に更新できる
        with st.status("AIがリクエストを分析中...", expanded=True) as status:
            
            # LangGraphバックエンドの呼び出し設定
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            input_data = {"messages": [HumanMessage(content=user_input)]}
            
            # 最終的に表示するコンテンツを保持する変数
            final_content = None

            # .invoke()の代わりに.stream()を使い、リアルタイムでチャンクを処理
            for chunk in compiled_workflow.stream(input_data, config):
                
                # --- chunkを解析して、実行状況をリアルタイムで表示 ---
                if "supervisor" in chunk:
                    # スーパーバイザーが思考中
                    status.update(label="どの専門家に依頼するか思考中...")
                elif "sql_node" in chunk:
                    # SQLノードが実行中
                    status.update(label="データベースにアクセスし、SQLを実行中...")
                elif "processing_node" in chunk:
                    # データ加工/グラフ作成ノードが実行中
                    status.update(label="データを加工し、グラフを作成中...")
                elif "interpret_node" in chunk:
                    # 解釈ノードが実行中
                    status.update(label="結果を解釈し、説明を生成中...")

                # ストリームの最後には、最終状態が'__end__'キーで含まれる
                if "__end__" in chunk:
                    final_state = chunk["__end__"]
                    final_content = final_state["messages"][-1]
            
            # --- 最終的な応答を解析して表示 ---

            # statusを「完了」状態に更新
            status.update(label="分析完了！", state="complete", expanded=False)
            
            # 表示用のコンテンツを格納する辞書
            display_content = {}
            
            # 返答の形式に応じて内容を解析
            if isinstance(final_content, AIMessage):
                # 通常のテキスト応答
                display_content["interpretation"] = final_content.content
                st.markdown(display_content["interpretation"])

            elif hasattr(final_content, 'content') and isinstance(final_content.content, str):
                try:
                    # JSON形式の文字列（データフレームやグラフ）
                    parsed_content = json.loads(final_content.content)
                    
                    if parsed_content.get("result_df_json"):
                        df_data = json.loads(parsed_content["result_df_json"])
                        display_content["result_df_json"] = parsed_content["result_df_json"]
                        st.dataframe(pd.DataFrame(df_data))
                    
                    if parsed_content.get("fig_json"):
                        # ここでは直接表示。履歴保存用にfig_jsonも保持
                        fig = pd.io.json.read_json(parsed_content["fig_json"], typ='frame')
                        display_content["fig_json"] = parsed_content["fig_json"]
                        st.plotly_chart(fig)

                except json.JSONDecodeError:
                    # JSONではないただのテキスト応答
                    display_content["interpretation"] = final_content.content
                    st.markdown(display_content["interpretation"])

            # 解析したAIの応答を履歴に追加
            if display_content:
                st.session_state.messages.append({"role": "assistant", "content": display_content})
