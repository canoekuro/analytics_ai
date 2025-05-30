import streamlit as st
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import time
from files.backend_codes import build_workflow

compiled_workflow = build_workflow()

# --- 1. セッション状態で履歴＆state管理 ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "memory_state" not in st.session_state:
    st.session_state["memory_state"] = {}

# --- 2. ワークフローのセットアップ（必要ならキャッシュ） ---
@st.cache_resource
def get_workflow():
    return build_workflow()

compiled_workflow = get_workflow()

# --- 3. チャットUI ---
st.title("SQL生成&データ解釈チャットAI")

# 3.1 履歴の表示（直近10件まで、または全件）
for entry in st.session_state["chat_history"]:
    role = entry["role"]
    if role == "user":
        st.chat_message("user").write(entry["content"])
    else:
        # assistant応答は「interpretation/普通テキスト」「df（データフレーム）」「chart_result（画像）」の3パターン
        with st.chat_message("assistant"):
            if "interpretation" in entry:
                st.write(entry["interpretation"])
            if "latest_df" in entry and entry["latest_df"]:
                df_disp = pd.DataFrame(entry["latest_df"])
                st.dataframe(df_disp)
            if "chart_result" in entry and entry["chart_result"]:
                chart_img = base64.b64decode(entry["chart_result"])
                st.image(chart_img, caption="AI生成グラフ")

# 3.2 ユーザー入力受付
user_input = st.chat_input("質問を入力してください（例: 'カテゴリごとの合計販売金額を出して'）")
if user_input:
    # --- 4. ユーザー入力バブルを即時表示 ---
    st.chat_message("user").write(user_input)
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    # --- 5. AIバブル(typing演出) ---
    ai_msg_placeholder = st.empty()
    for i in range(8):  # 約4秒間タイピング演出
        dots = "." * ((i % 4) + 1)
        ai_msg_placeholder.chat_message("assistant").write(f"AI is thinking{dots} _typing_ :speech_balloon:")
        time.sleep(0.5)

    # --- 6. LangGraphバックエンド呼び出し ---
    config = {'configurable': {'thread_id': '1'}}
    input_state = dict(st.session_state.get("memory_state", {}))
    input_state["input"] = user_input
    res = compiled_workflow.invoke(input_state, config)

    # --- 7. AIバブル差し替え ---
    with ai_msg_placeholder.chat_message("assistant"):
        if "interpretation" in res:
            st.write(res["interpretation"])
        if "latest_df" in res and res["latest_df"]:
            st.dataframe(pd.DataFrame(res["latest_df"]))
        if "chart_result" in res and res["chart_result"]:
            st.image(base64.b64decode(res["chart_result"]), caption="AI生成グラフ")
    # --- 8. 履歴に保存 ---
    st.session_state["chat_history"].append({"role": "assistant", **res})
    st.session_state["memory_state"] = res
