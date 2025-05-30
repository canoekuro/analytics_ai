# market_analysis
市場分析用のAIを作成します。

1. システム概要

ユーザーの自然言語入力に対して、LangGraph（Azure OpenAI, RAG, memory付き）を用いて
・SQL生成・実行・データ抽出
・データフレーム表示
・グラフ化
・解釈コメント
を一気通貫で返す「対話型データ分析チャット」アプリを開発する。
フロントエンドはStreamlit、データベースはSQLite。
チャット履歴や会話文脈はLangGraphのmemory機能で管理。

2. システム要件

2.1 コア機能

・ユーザー自然言語からのSQL生成、実行、DataFrame抽出
・複数回のDataFrame抽出、解釈、グラフ生成と、その結果のステップ間での引継ぎ
・DataFrameからの自然言語による解釈
・DataFrameのグラフ自動生成・画像として表示
・テーブル定義・クエリ例RAGによる高精度SQL生成
・StreamlitチャットUIによる自然な会話体験（履歴はバブル形式で表示）

2.2 チャット体験要件

・会話履歴（チャット履歴）はLangGraphのmemory機能で保持
  各ノード（SQL/グラフ/解釈など）で過去のやり取りや一時的なデータ（df, chart等）を自動参照
  ユーザーが「前のデータをグラフ化して」等の文脈指示にも自然に対応
・Streamlitではst.chat_input, st.chat_message、st.dataframe, st.image等を活用してチャット体験を最大化
・データフレームはst.dataframeで見やすく表示
・グラフは画像（base64デコード）で表示
・解釈コメントは普通のテキストとしてそのまま返す
・ローディングインジケータやストリーミング出力（typing...演出）にも対応

2.3 データベース

・SQLite（ローカル or クラウド環境で動作）

3. アーキテクチャ

3.1 フロントエンド（Streamlit）

・チャットUIとしてStreamlitを使用（履歴管理はst.session_state＋LangGraphのmemory）
・入力ごとにstateとmemoryをLangGraphに渡し、出力を受け取り表示

3.2 バックエンド（LangGraph）

・LangGraphワークフロー＋state（AI各ノード間でデータ・履歴を一貫管理）
・memory機能により「会話履歴」「一時的な生成物」を保持し、ノード間・ターン間での文脈を維持
・RAGベースのテーブル定義・クエリ例のベクトルストア（FAISS）
・Azure OpenAI（GPT-4o/mini）を利用

3.3 データベース

・SQLite（pandas DataFrameとして取得・可視化）

4. ワークフロー例

1. ユーザー：「カテゴリの合計販売金額を出して」
   → SQL生成＆実行、DataFrame取得、DataFrame表示、解釈コメント
2. ユーザー：「このデータをグラフ化して」
   → memoryから直前のdfを自動参照、グラフ生成・表示
3. ユーザー：「この傾向について解釈して」
   → memoryから直前のdfやグラフを自動参照、解釈コメントを返す

5. state/memory設計

・state（dict）… 1ターンごとの入力・出力を保持（input, df, SQL, chart_result, interpretation等）
・memory（LangGraphのMessageHistory等）… 全チャット履歴・一時生成物を自動保持
  各ノード関数内でstate.get("memory")を参照し、プロンプト組み立てやデータ再利用が可能
・Streamlitのst.session_stateでもstateとmemoryを持ち、ユーザーのセッション中は常に維持

6. 開発・実装方針

・バックエンド（LangGraph）のワークフロー・ノード群はworkflow_backend.py等にモジュール分離
・フロントエンド（Streamlit）はapp.pyで、stateとmemoryをやり取り
・AI応答の出力（DataFrame・グラフ画像・テキスト）はバブルUIとして順次表示
