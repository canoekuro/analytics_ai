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
user_input = st.chat_input("分析したいことを入力してください（例: カテゴリ別の売上を見せて）")

if user_input:
    # ユーザーの入力を履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AIの応答を待つ間、スピナー（くるくる回るアイコン）を表示
    with st.spinner("AIが分析・考察中です..."):
        try:
            # LangGraphバックエンドの呼び出し
            # セッションごとにユニークなthread_idを設定
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            
            # LangGraphが期待する入力形式に合わせる
            input_data = {"messages": [HumanMessage(content=user_input)]}
            
            # ワークフローを実行
            final_state = compiled_workflow.invoke(input_data, config)
                        # 応答メッセージリストを取得
            response_messages = final_state["messages"]
            ai_response_to_display = None

            # メッセージを末尾から遡って、ユーザーに表示すべき「本当の応答」を探す
            for i in range(len(response_messages) - 1, -1, -1):
                msg = response_messages[i]
                
                # スーパーバイザーの最終判断（DispatchDecision）に関連するメッセージはスキップ
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    if msg.tool_calls[0]['name'] == 'DispatchDecision':
                        continue
                if isinstance(msg, HumanMessage):
                    # 人間のメッセージまで遡ったら探索終了
                    break

                # スキップされなかった最初のAIまたはToolメッセージが、表示すべき応答
                ai_response = msg
                break
            
            # もし見つからなければ、念のため最後のメッセージを使う
            if ai_response_to_display is None:
                ai_response_to_display = response_messages[-1]

            
            response_content = None
            
            # 返答の形式に応じて内容を解析
            if isinstance(ai_response, AIMessage):
                # 通常のテキスト応答
                 response_content = {"interpretation": ai_response.content}

            elif hasattr(ai_response, 'content') and isinstance(ai_response.content, str):
                 try:
                    # JSON形式の文字列（データフレームやグラフ）
                    parsed_content = json.loads(ai_response.content)
                    
                    # ToolMessageから返ってくるデータはさらにネストしていることがあるため、中身を取り出す
                    df_json = parsed_content.get("result_df_json")
                    fig_json = parsed_content.get("fig_json")
                    
                    response_content = {
                        "result_df_json": df_json,
                        "fig_json": fig_json
                    }

                 except json.JSONDecodeError:
                    # JSONではないただのテキスト応答
                    response_content = {"interpretation": ai_response.content}

            # 解析したAIの応答を履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            
            # 画面を再読み込みして、最新のチャット履歴を表示
            st.rerun()

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
