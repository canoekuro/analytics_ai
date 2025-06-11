import re
import sqlite3
import pandas as pd
import logging
from typing import TypedDict, List, Optional, Any
import ast

#SQL関連の関数
#SQLのコードブロックがあった際にそれを削除
def extract_sql(sql_text):
    match = re.search(r"```sql\s*(.*?)```", sql_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", sql_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return sql_text.strip()

#SQLのアウトプットを必ずList[str]として後続処理に流す。
def parse_llm_list_output(llm_output_str: str) -> List[str]:
    try:
        cleaned_str = re.sub(r'^\s*[-*]\s*', '', llm_output_str, flags=re.MULTILINE)
        if not cleaned_str.strip().startswith('['):
            lines = [line.strip().replace("'", "\\'") for line in cleaned_str.split('\n') if line.strip()]
            if all(not (line.startswith("'") and line.endswith("'")) and \
                   not (line.startswith('"') and line.endswith('"')) for line in lines):
                 cleaned_str = "[" + ", ".join([f"'{line}'" for line in lines]) + "]"
            else:
                 cleaned_str = "[" + ", ".join(lines) + "]"
        parsed_list = ast.literal_eval(cleaned_str)
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            return parsed_list
        return []
    except (ValueError, SyntaxError) as e:
        logging.warning(f"ast.literal_evalによるLLMリスト出力の解析に失敗しました: {e}。元の文字列: '{llm_output_str}'。改行による分割にフォールバックします。")
        str_for_split = llm_output_str.strip()
        if str_for_split.startswith('[') and str_for_split.endswith(']'):
            str_for_split = str_for_split[1:-1]
        suggestions = []
        for line in str_for_split.split('\n'):
            item = re.sub(r'^\s*[-*]\s*', '', line).strip()
            item = item.strip('\'",')
            if item:
                 suggestions.append(item)
        return suggestions

#SQLを実行
def try_sql_execute(sql_text):
    try:
        conn = sqlite3.connect("my_data.db")
        df = pd.read_sql(sql_text, conn)
        return df, None
    except Exception as e:
        logging.error(f"SQLの実行に失敗しました。エラー: {e}\nSQL: {sql_text}", exc_info=True)
        return None, str(e)

#SQLのエラーがあった際にエラー内容を日本語にする
def transform_sql_error(sql_error: str) -> str:
    if "no such table" in sql_error.lower():
        return f"指定されたテーブルが見つからなかったため、クエリを実行できませんでした。テーブル名を確認してください。(詳細: {sql_error})"
    elif "no such column" in sql_error.lower():
        return f"指定された列が見つからなかったため、クエリを実行できませんでした。列名を確認してください。(詳細: {sql_error})"
    elif "syntax error" in sql_error.lower():
        return f"SQL構文にエラーがあります。クエリを確認してください。(詳細: {sql_error})"
    else:
        return f"SQLクエリの処理中に予期せぬエラーが発生しました。管理者に連絡してください。(詳細: {sql_error})"

#エラー内容に沿ってSQLを修正
def fix_sql_with_llm(llm, original_sql, error_message, rag_tables, rag_queries, task_description, context):
    prompt = f"""
    以下はユーザーの質問・関連情報・AIが生成したSQLとその実行時のエラー内容です。
    エラー内容を踏まえて、SQLを修正してください。
    SQLのみ出力し、前後のコメントや説明文は不要です。

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【テーブル定義に関する情報】
    {rag_tables}

    【類似する問い合わせ例とそのSQL】
    {rag_queries}

    【生成SQL】
    {original_sql}

    【エラー内容】
    {error_message}
    """
    response = llm.invoke(prompt)
    fixed_sql = extract_sql(response.content)
    return fixed_sql

def get_table_name_from_formatted_doc(doc_content: str) -> Optional[str]:
    for line in doc_content.split('\n'):
        line = line.strip()
        if line.startswith("<table_name>") and line.endswith("</table_name>"):
            return line[len("<table_name>"):-len("</table_name>")].strip()
    return None  # for全部回して見つからなかったらNone
