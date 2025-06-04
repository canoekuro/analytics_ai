import logging
from files.backend_codes import build_workflow
from langchain_core.messages import HumanMessage


# 基本的なロギングを設定
logging.basicConfig(level=logging.INFO)

from langchain_core.messages import HumanMessage
logging.basicConfig(level=logging.INFO)
workflow = build_workflow()

# 例：ユーザー入力から初期状態を作って実行

state = {
    "messages": [
        HumanMessage(content="利用可能なデータの内容が知りたい。")
    ],
    "next": "supervisor",
    "df_history": []
}
result = workflow.invoke(state, config={"thread_id": "my_test_session_001","return_intermediate_steps": True})
print(result)
