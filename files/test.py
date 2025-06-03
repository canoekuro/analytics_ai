import logging
from files.backend_codes import build_workflow

# 基本的なロギングを設定
logging.basicConfig(level=logging.INFO)

from langchain_core.messages import HumanMessage
logging.basicConfig(level=logging.INFO)
workflow = build_workflow()

# 例：ユーザー入力から初期状態を作って実行
state = {
    "messages": [HumanMessage(content="データベース内のテーブルの一覧を取得したい")],
    "task_description": [],
    "metadata_answer": [],
    "df_history": [],
    "sql_history": [],
    "interpretation_history": [],
    "chart_history": [],
    "analize_step": []
}

result = workflow.invoke(state, config={"thread_id": "my_test_session_001"})
print(result)