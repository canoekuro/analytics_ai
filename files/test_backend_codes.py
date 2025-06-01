import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import difflib # Added for mocking SequenceMatcher
import sqlite3 # For testing try_sql_execute with in-memory DB
import collections
import uuid
from datetime import datetime

# Assuming backend_codes.py is in the same directory or accessible via PYTHONPATH
from files.backend_codes import (
    MyState,
    classify_intent_node,
    extract_data_requirements_node,
    find_similar_query_node,
    sql_node,
    build_workflow,
    try_sql_execute, # Import for direct testing
    # For mocking, we might need to patch where they are *used* if not careful with direct imports
    # However, for llm, vectorstores, these are global in backend_codes, so patching them there is fine.
)

# Mocking global objects from backend_codes.py
# These will be patched in specific test methods or setUp using @patch
# llm_mock = MagicMock()
# vectorstore_tables_mock = MagicMock()
# vectorstore_queries_mock = MagicMock()
# try_sql_execute_mock = MagicMock()


class TestExtractDataRequirementsNode(unittest.TestCase):

    @patch('files.backend_codes.llm')
    def test_extract_simple_input(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="売上データ")
        state = MyState(input="売上データを見せて", intent_list=["データ取得"])

        result_state = extract_data_requirements_node(state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["data_requirements"], ["売上データ"])
        self.assertEqual(result_state["condition"], "データ要件抽出完了")

    @patch('files.backend_codes.llm')
    def test_extract_complex_input_multiple_requirements(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="A商品の売上集計,顧客属性データ")
        state = MyState(input="A商品の売上集計とお客様属性のクロス集計グラフを出して", intent_list=["データ取得", "グラフ作成"])

        result_state = extract_data_requirements_node(state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["data_requirements"], ["A商品の売上集計", "顧客属性データ"])
        self.assertEqual(result_state["condition"], "データ要件抽出完了")

    @patch('files.backend_codes.llm')
    def test_extract_input_no_clear_requirements(self, mock_llm):
        # LLM might return empty or a generic phrase. Let's assume empty for this test.
        mock_llm.invoke.return_value = MagicMock(content="")
        state = MyState(input="こんにちは、調子どう？", intent_list=[]) # No data intent

        result_state = extract_data_requirements_node(state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["data_requirements"], [])
        self.assertEqual(result_state["condition"], "データ要件抽出完了")

    @patch('files.backend_codes.llm')
    def test_extract_input_llm_returns_whitespace_and_commas(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content=" 要件1 ,, 要件2 , ")
        state = MyState(input="何かデータちょうだい", intent_list=["データ取得"])

        result_state = extract_data_requirements_node(state)

        self.assertEqual(result_state["data_requirements"], ["要件1", "要件2"])
        self.assertEqual(result_state["condition"], "データ要件抽出完了")


class TestFindSimilarQueryNode(unittest.TestCase):

    def test_all_requirements_found(self):
        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の売上集計", "timestamp": "ts1",
            "dataframe_dict": [{"product": "A", "sales": 100}], "SQL": "SQL1"
        }
        sample_history_entry_2 = {
            "id": "hist_002", "query": "顧客属性データ", "timestamp": "ts2",
            "dataframe_dict": [{"user_id": 1, "age": 30}], "SQL": "SQL2"
        }
        state = MyState(
            input="A商品の売上集計と顧客属性データが欲しい",
            data_requirements=["A商品の売上集計", "顧客属性データ"],
            df_history=[sample_history_entry_1, sample_history_entry_2],
            intent_list=["データ取得"]
        )

        result_state = find_similar_query_node(state)

        self.assertEqual(result_state["condition"], "all_data_found")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 2)
        self.assertEqual(result_state["latest_df"]["A商品の売上集計"], [{"product": "A", "sales": 100}])
        self.assertEqual(result_state["latest_df"]["顧客属性データ"], [{"user_id": 1, "age": 30}])
        self.assertEqual(result_state["missing_data_requirements"], [])

    @patch('files.backend_codes.SIMILARITY_THRESHOLD', 0.8) # Fix threshold for this test
    @patch('files.backend_codes.difflib.SequenceMatcher')
    def test_some_requirements_missing(self, MockSequenceMatcher, mock_threshold):
        # Configure the mock instance that will be created inside find_similar_query_node
        mock_matcher_instance = MockSequenceMatcher.return_value

        # This list will provide return values for successive calls to ratio().
        # Order:
        # 1. Req "A商品の売上" vs Hist "A商品の売上集計" -> Found
        # 2. Req "B商品の在庫" vs Hist "A商品の売上集計" -> Not Found
        mock_matcher_instance.ratio.side_effect = [
            0.9, # Above 0.8 threshold
            0.5  # Below 0.8 threshold
        ]

        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の売上集計", "timestamp": "ts1",
            "dataframe_dict": [{"product": "A", "sales": 100}], "SQL": "SQL1"
        }
        state = MyState(
            input="A商品の売上とB商品の在庫データが欲しい",
            data_requirements=["A商品の売上", "B商品の在庫"],
            df_history=[sample_history_entry_1],
            intent_list=["データ取得"]
        )

        result_state = find_similar_query_node(state)

        self.assertEqual(result_state["condition"], "missing_data")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 1)
        self.assertTrue("A商品の売上" in result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["A商品の売上"], [{"product": "A", "sales": 100}])
        self.assertEqual(result_state["missing_data_requirements"], ["B商品の在庫"])

    def test_none_requirements_found(self):
        sample_history_entry_1 = {
            "id": "hist_001", "query": "X商品の情報", "timestamp": "ts1",
            "dataframe_dict": [{"product": "X", "info": "xyz"}], "SQL": "SQLX"
        }
        state = MyState(
            input="A商品の売上とB商品の在庫データが欲しい",
            data_requirements=["A商品の売上", "B商品の在庫"],
            df_history=[sample_history_entry_1],
            intent_list=["データ取得"]
        )

        result_state = find_similar_query_node(state)

        self.assertEqual(result_state["condition"], "missing_data")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 0)
        self.assertEqual(len(result_state["missing_data_requirements"]), 2)
        self.assertIn("A商品の売上", result_state["missing_data_requirements"])
        self.assertIn("B商品の在庫", result_state["missing_data_requirements"])

    def test_empty_data_requirements(self):
        state = MyState(
            input="何かグラフにして",
            data_requirements=[], # Explicitly empty
            df_history=[],
            intent_list=["グラフ作成"] # No "データ取得" intent
        )

        result_state = find_similar_query_node(state)

        self.assertEqual(result_state["condition"], "no_requirements_specified")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(result_state["latest_df"], collections.OrderedDict()) # Check for empty OrderedDict
        self.assertEqual(result_state["missing_data_requirements"], [])

    @patch('files.backend_codes.SIMILARITY_THRESHOLD', 0.8) # Fix threshold for this test
    @patch('files.backend_codes.difflib.SequenceMatcher')
    def test_history_entry_used_once(self, MockSequenceMatcher, mock_threshold):
        mock_matcher_instance = MockSequenceMatcher.return_value

        # Scenario:
        # Req 1: "A商品の詳細"
        # Req 2: "A商品の情報"
        # History 1: "A商品の詳細データ"
        #
        # Comparisons made by find_similar_query_node (order matters):
        # Loop 1 (req "A商品の詳細"):
        #   - "A商品の詳細" vs "A商品の詳細データ" -> ratio() call 1
        # Loop 2 (req "A商品の情報"):
        #   - "A商品の情報" vs "A商品の詳細データ" -> ratio() call 2 (if history not used yet)
        #
        # We want "A商品の詳細" to match "A商品の詳細データ" strongly.
        # We want "A商品の情報" to match "A商品の詳細データ" less strongly (or below threshold),
        # or ensure that if it does match, the history entry is already used.
        # Let's make "A商品の詳細" the clear winner and ensure it uses the history.
        # Then "A商品の情報" will find the history entry already used.

        mock_matcher_instance.ratio.side_effect = [
            0.95, # "A商品の詳細" vs "A商品の詳細データ" (high, above threshold)
            0.85  # "A商品の情報" vs "A商品の詳細データ" (also above, but history will be used)
                  # This value doesn't strictly matter if the history is correctly marked as used.
                  # If it were below threshold, that would also work for this test's logic.
        ]

        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の詳細データ", "timestamp": "ts1",
            "dataframe_dict": [{"product": "A", "detail": "very detailed"}], "SQL": "SQL_A_detail"
        }
        state = MyState(
            input="A商品の詳細とA商品の情報",
            data_requirements=["A商品の詳細", "A商品の情報"], # Order of requirements processing
            df_history=[sample_history_entry_1],
            intent_list=["データ取得"]
        )

        result_state = find_similar_query_node(state)

        self.assertEqual(result_state["condition"], "missing_data")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 1)

        # "A商品の詳細" should be found because it's processed first and has a high similarity
        self.assertIn("A商品の詳細", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["A商品の詳細"], [{"product": "A", "detail": "very detailed"}])
        self.assertIn("A商品の情報", result_state["missing_data_requirements"])
        self.assertNotIn("A商品の情報", result_state["latest_df"])


class TestSqlNode(unittest.TestCase):

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4') # To control history IDs
    def test_sql_node_with_missing_requirements(
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        # This test does not directly depend on SIMILARITY_THRESHOLD, so no changes needed for that.
        # Keeping it here just to show its position relative to the modified tests.
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="test_uuid_".ljust(8, '0'))) # Ensure 8 chars for [:8] slice

        # --- Mocking RAG results ---
        mock_vs_tables.return_value = [MagicMock(page_content="Table info for Sales")]
        mock_vs_queries.return_value = [MagicMock(page_content="Query info for Sales")]

        # --- Mocking LLM responses for SQL generation ---
        # LLM called once for "Sales Data", once for "Inventory Data"
        mock_llm.invoke.side_effect = [
            MagicMock(content="SELECT * FROM sales;"), # For "Sales Data"
            MagicMock(content="SELECT * FROM inventory;")  # For "Inventory Data"
        ]

        # --- Mocking DB execution results ---
        # try_sql_execute called once for sales, once for inventory
        sales_df = pd.DataFrame([{"item": "Laptop", "qty": 10}])
        inventory_df = pd.DataFrame([{"item": "Laptop", "stock": 50}])
        mock_try_sql.side_effect = [
            (sales_df, None),       # Successful execution for sales SQL
            (inventory_df, None)    # Successful execution for inventory SQL
        ]

        initial_state = MyState(
            input="Show me sales and inventory.",
            intent_list=["データ取得"],
            data_requirements=["Sales Data", "Inventory Data"], # Assume these were extracted
            missing_data_requirements=["Sales Data", "Inventory Data"], # These are what sql_node will process
            latest_df=collections.OrderedDict(), # Initially empty
            df_history=[]
        )

        result_state = sql_node(initial_state)

        # --- Assertions ---
        self.assertEqual(result_state["condition"], "SQL実行完了")
        self.assertIsNone(result_state["error"])

        # Check latest_df
        self.assertIn("Sales Data", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["Sales Data"], sales_df.to_dict(orient="records"))
        self.assertIn("Inventory Data", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["Inventory Data"], inventory_df.to_dict(orient="records"))

        # Check df_history
        self.assertEqual(len(result_state["df_history"]), 2)
        history_sales = next(h for h in result_state["df_history"] if h["query"] == "Sales Data")
        history_inventory = next(h for h in result_state["df_history"] if h["query"] == "Inventory Data")

        self.assertEqual(history_sales["SQL"], "SELECT * FROM sales;")
        self.assertEqual(history_inventory["SQL"], "SELECT * FROM inventory;")

        # Check calls to mocks
        self.assertEqual(mock_vs_tables.call_count, 2)
        mock_vs_tables.assert_any_call("Sales Data", k=3)
        mock_vs_tables.assert_any_call("Inventory Data", k=3)

        self.assertEqual(mock_vs_queries.call_count, 2)
        self.assertEqual(mock_llm.invoke.call_count, 2)
        self.assertEqual(mock_try_sql.call_count, 2)
        mock_try_sql.assert_any_call("SELECT * FROM sales;")
        mock_try_sql.assert_any_call("SELECT * FROM inventory;")

        # Check if missing_data_requirements is updated (cleared in this case)
        self.assertEqual(len(result_state.get("missing_data_requirements", [])), 0)


    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4')
    def test_sql_node_fallback_behavior(
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="test_uuid_fallback".ljust(8, '0')))
        mock_vs_tables.return_value = [MagicMock(page_content="General Table Info")]
        mock_vs_queries.return_value = [MagicMock(page_content="General Query Info")]
        mock_llm.invoke.return_value = MagicMock(content="SELECT * FROM general_query;")

        general_df = pd.DataFrame([{"info": "general data"}])
        mock_try_sql.return_value = (general_df, None)

        initial_state = MyState(
            input="Show me general data.",
            intent_list=["データ取得"],
            data_requirements=[], # No specific requirements extracted, or not applicable
            missing_data_requirements=[], # Empty, so should trigger fallback
            latest_df=collections.OrderedDict(),
            df_history=[]
        )

        result_state = sql_node(initial_state)

        self.assertEqual(result_state["condition"], "SQL実行完了")
        self.assertIsNone(result_state["error"])

        # latest_df should contain the data keyed by the original input query
        self.assertIn("Show me general data.", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["Show me general data."], general_df.to_dict(orient="records"))

        # df_history should reflect the general query
        self.assertEqual(len(result_state["df_history"]), 1)
        history_entry = result_state["df_history"][0]
        self.assertEqual(history_entry["query"], "Show me general data.")
        self.assertEqual(history_entry["SQL"], "SELECT * FROM general_query;")

        mock_vs_tables.assert_called_once_with("Show me general data.", k=3)
        mock_llm.invoke.assert_called_once()
        mock_try_sql.assert_called_once_with("SELECT * FROM general_query;")

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4')
    def test_sql_node_partial_failure_for_missing_requirements(
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="test_uuid_partial".ljust(8, '0')))
        mock_vs_tables.return_value = [MagicMock(page_content="Info")] # Generic RAG
        mock_vs_queries.return_value = [MagicMock(page_content="Query Info")]

        # This sequence for llm.invoke:
        # 1. SQL generation for "Success Req"
        # 2. SQL generation for "Error Req" (initial attempt)
        # 3. SQL generation by fix_sql_with_llm for "Error Req" (after first failure)
        mock_llm.invoke.side_effect = [
            MagicMock(content="SELECT * FROM success_req;"),
            MagicMock(content="SELECT * FROM error_req_initial;"),
            MagicMock(content="SELECT * FROM error_req_fixed;")
        ]

        success_df = pd.DataFrame([{"data": "success"}])
        # This sequence for try_sql_execute:
        # 1. "Success Req" - success
        # 2. "Error Req" (initial SQL) - fails
        # 3. "Error Req" (fixed SQL) - fails again
        mock_try_sql.side_effect = [
            (success_df, None),
            (None, "Initial SQL error"),
            (None, "Persistent SQL error")
        ]

        initial_state = MyState(
            input="Get success and error data.",
            missing_data_requirements=["Success Req", "Error Req"],
            latest_df=collections.OrderedDict(),
            df_history=[]
        )

        result_state = sql_node(initial_state)

        self.assertEqual(result_state["condition"], "SQL部分的失敗")
        self.assertIsNotNone(result_state["error"])
        self.assertIn("Failed to get data for 'Error Req': Persistent SQL error", result_state["error"])

        self.assertIn("Success Req", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["Success Req"], success_df.to_dict(orient="records"))
        self.assertNotIn("Error Req", result_state["latest_df"])

        self.assertEqual(len(result_state["df_history"]), 1)
        self.assertEqual(result_state["df_history"][0]["query"], "Success Req")

        self.assertEqual(mock_llm.invoke.call_count, 3)
        self.assertEqual(mock_try_sql.call_count, 3)

        self.assertEqual(result_state.get("missing_data_requirements"), ["Error Req"])


class TestSqlNodeCorrection(unittest.TestCase):

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4')
    def test_sql_correction_successful(
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="corr_ok_".ljust(8, '0')))
        req_string = "Sales for Product X"
        initial_sql = "SELECT sale FROM productX_sales;" # Intentionally problematic
        fixed_sql = "SELECT sales FROM product_X_sales;" # Corrected SQL
        expected_df_data = [{"sales": 100}]
        expected_df = pd.DataFrame(expected_df_data)

        # RAG mocks (called for initial SQL gen and for fix_sql_with_llm)
        mock_vs_tables.return_value = [MagicMock(page_content="Table: product_X_sales (sales, product_id)")]
        mock_vs_queries.return_value = [MagicMock(page_content="SELECT sales FROM product_X_sales WHERE product_id = 'X'")]

        # LLM sequence:
        # 1. Initial SQL generation
        # 2. fix_sql_with_llm (receives error, generates fixed_sql)
        mock_llm.invoke.side_effect = [
            MagicMock(content=initial_sql), # Initial SQL generation
            MagicMock(content=fixed_sql)    # SQL generated by fix_sql_with_llm
        ]

        # DB execution sequence:
        # 1. Initial SQL fails
        # 2. Fixed SQL succeeds
        mock_try_sql.side_effect = [
            (None, "no such column: sale"), # Error for initial_sql
            (expected_df, None)             # Success for fixed_sql
        ]

        initial_state = MyState(
            input=f"Get {req_string}",
            missing_data_requirements=[req_string],
            latest_df=collections.OrderedDict(),
            df_history=[]
        )

        result_state = sql_node(initial_state)

        self.assertEqual(result_state["condition"], "SQL実行完了")
        self.assertIsNone(result_state["error"])
        self.assertIn(req_string, result_state["latest_df"])
        self.assertEqual(result_state["latest_df"][req_string], expected_df_data)
        self.assertEqual(result_state["SQL"], fixed_sql) # Should store the successfully executed SQL

        self.assertEqual(mock_llm.invoke.call_count, 2)
        # Call 1: Initial SQL generation
        self.assertIn(req_string, str(mock_llm.invoke.call_args_list[0].args[0]))
        # Call 2: fix_sql_with_llm
        fixer_prompt_args = mock_llm.invoke.call_args_list[1].args[0]
        self.assertIn(initial_sql, str(fixer_prompt_args))
        self.assertIn("no such column: sale", str(fixer_prompt_args))
        self.assertIn(req_string, str(fixer_prompt_args)) # User query context

        self.assertEqual(mock_try_sql.call_count, 2)
        mock_try_sql.assert_any_call(initial_sql)
        mock_try_sql.assert_any_call(fixed_sql)

        self.assertEqual(len(result_state.get("missing_data_requirements", [])), 0)

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4') # Not strictly needed here but good practice if history was added
    @patch('files.backend_codes.transform_sql_error') # Mock the error transformation
    def test_sql_correction_fails_again(
        self, mock_transform_sql_error, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="corr_fail".ljust(8, '0')))
        req_string = "Inventory for Product Y"
        initial_sql = "SELECT inventry FROM productY_inventory;"
        fixed_sql_attempt = "SELECT inventory FROM product_Y_inventory_typo;" # Still wrong

        # Mock RAG
        mock_vs_tables.return_value = [MagicMock(page_content="Table: product_Y_inventory (inventory, product_id)")]
        mock_vs_queries.return_value = [MagicMock(page_content="SELECT inventory FROM product_Y_inventory")]

        # LLM sequence:
        mock_llm.invoke.side_effect = [
            MagicMock(content=initial_sql),
            MagicMock(content=fixed_sql_attempt)
        ]

        # DB execution sequence: both fail
        mock_try_sql.side_effect = [
            (None, "no such column: inventry"),
            (None, "no such table: product_Y_inventory_typo")
        ]

        # Mock the error transformation to check it's called with the final error
        mock_transform_sql_error.return_value = "User-friendly: Table not found"


        initial_state = MyState(
            input=f"Get {req_string}",
            missing_data_requirements=[req_string],
            latest_df=collections.OrderedDict(),
            df_history=[]
        )

        result_state = sql_node(initial_state)

        self.assertEqual(result_state["condition"], "SQL部分的失敗") # Or "SQL実行失敗" if only one req
        self.assertIsNotNone(result_state["error"])
        # Check that the user-friendly message is in the error string
        self.assertIn("User-friendly: Table not found", result_state["error"])
        self.assertNotIn(req_string, result_state["latest_df"]) # No data should be added
        self.assertEqual(result_state["SQL"], fixed_sql_attempt) # Stores the last attempted SQL

        self.assertEqual(mock_llm.invoke.call_count, 2)
        self.assertEqual(mock_try_sql.call_count, 2)
        mock_transform_sql_error.assert_called_once_with("no such table: product_Y_inventory_typo")
        self.assertEqual(result_state.get("missing_data_requirements"), [req_string])


class TestTrySqlExecute(unittest.TestCase):

    @patch('files.backend_codes.sqlite3.connect')
    def test_try_sql_execute_success_in_memory(self, mock_sqlite_connect):
        # Setup in-memory database
        in_memory_conn = sqlite3.connect(':memory:')
        mock_sqlite_connect.return_value = in_memory_conn

        cursor = in_memory_conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        cursor.execute("INSERT INTO test_table VALUES (1, 'Test User One')")
        cursor.execute("INSERT INTO test_table VALUES (2, 'Test User Two')")
        in_memory_conn.commit()

        df, error = try_sql_execute("SELECT id, name FROM test_table ORDER BY id ASC")

        self.assertIsNotNone(df)
        self.assertIsNone(error)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['name'], 'Test User One')
        self.assertEqual(df.iloc[1]['id'], 2)

        # try_sql_execute internally calls connect("my_data.db")
        mock_sqlite_connect.assert_called_once_with("my_data.db")
        in_memory_conn.close()

    @patch('files.backend_codes.sqlite3.connect')
    def test_try_sql_execute_error_in_memory(self, mock_sqlite_connect):
        # Setup in-memory database
        in_memory_conn = sqlite3.connect(':memory:')
        mock_sqlite_connect.return_value = in_memory_conn

        # No table created, so SELECT will fail

        df, error = try_sql_execute("SELECT * FROM non_existent_table")

        self.assertIsNone(df)
        self.assertIsNotNone(error)
        self.assertIn("no such table: non_existent_table", error.lower())

        mock_sqlite_connect.assert_called_once_with("my_data.db")
        in_memory_conn.close()

    @patch('files.backend_codes.sqlite3.connect')
    def test_try_sql_execute_connection_failure(self, mock_sqlite_connect):
        # Simulate sqlite3.connect itself raising an error
        mock_sqlite_connect.side_effect = sqlite3.OperationalError("unable to open database file")

        df, error = try_sql_execute("SELECT * FROM any_table")

        self.assertIsNone(df)
        self.assertIsNotNone(error)
        self.assertIn("unable to open database file", error.lower())
        mock_sqlite_connect.assert_called_once_with("my_data.db")

    @patch('files.backend_codes.sqlite3.connect')
    def test_try_sql_execute_syntax_error(self, mock_sqlite_connect):
        # Setup in-memory database
        in_memory_conn = sqlite3.connect(':memory:')
        mock_sqlite_connect.return_value = in_memory_conn
        # No table needed, syntax error is independent of tables

        # Malformed SQL to cause a syntax error
        malformed_sql = "SELEC * FROM test_table"
        df, error = try_sql_execute(malformed_sql)

        self.assertIsNone(df)
        self.assertIsNotNone(error)
        # SQLite's error message for syntax errors often includes "syntax error"
        # and might point near the problematic token.
        self.assertIn("syntax error", error.lower())
        mock_sqlite_connect.assert_called_once_with("my_data.db")
        in_memory_conn.close()

    @patch('files.backend_codes.sqlite3.connect')
    def test_try_sql_execute_no_such_column(self, mock_sqlite_connect):
        # Setup in-memory database
        in_memory_conn = sqlite3.connect(':memory:')
        mock_sqlite_connect.return_value = in_memory_conn
        cursor = in_memory_conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        in_memory_conn.commit()

        # SQL that refers to a non-existent column
        sql_with_bad_column = "SELECT non_existent_column FROM test_table"
        df, error = try_sql_execute(sql_with_bad_column)

        self.assertIsNone(df)
        self.assertIsNotNone(error)
        # SQLite's error for a non-existent column typically includes "no such column"
        self.assertIn("no such column: non_existent_column", error.lower())
        mock_sqlite_connect.assert_called_once_with("my_data.db")
        in_memory_conn.close()


# Import functions to be tested
from files.backend_codes import (
    extract_sql,
    fix_sql_with_llm,
    transform_sql_error,
    clear_data_node,
    DATA_CLEARED_MESSAGE,
    interpret_node,
    chart_node,
    metadata_retrieval_node
)


class TestExtractSQL(unittest.TestCase):
    def test_extract_with_sql_language_identifier(self):
        text = "Some text before ```sql\nSELECT * FROM table;\n``` and after"
        self.assertEqual(extract_sql(text), "SELECT * FROM table;")

    def test_extract_with_sql_language_identifier_mixed_case(self):
        text = "```SQL\nSELECT column FROM test_table;\n```"
        self.assertEqual(extract_sql(text), "SELECT column FROM test_table;")

    def test_extract_without_language_identifier(self):
        text = "```\nSELECT name FROM users;\n```"
        self.assertEqual(extract_sql(text), "SELECT name FROM users;")

    def test_extract_with_leading_trailing_whitespace_in_sql(self):
        text = "```sql\n  SELECT id FROM products;  \n```"
        self.assertEqual(extract_sql(text), "SELECT id FROM products;")

    def test_extract_with_leading_trailing_whitespace_outside_backticks(self):
        text = "  ```sql\nSELECT id FROM products;\n```  "
        self.assertEqual(extract_sql(text), "SELECT id FROM products;")

    def test_plain_sql_string_no_backticks(self):
        text = "SELECT * FROM orders;"
        self.assertEqual(extract_sql(text), "SELECT * FROM orders;")

    def test_plain_sql_string_with_leading_trailing_whitespace(self):
        text = "  SELECT * FROM orders;  "
        self.assertEqual(extract_sql(text), "SELECT * FROM orders;")

    def test_string_with_backticks_but_no_sql_content(self):
        text = "```sql\n```"
        self.assertEqual(extract_sql(text), "")

        text_no_lang = "```\n```"
        self.assertEqual(extract_sql(text_no_lang), "")

    def test_string_with_backticks_and_only_whitespace(self):
        text = "```sql\n   \n```"
        self.assertEqual(extract_sql(text), "")

    def test_empty_string(self):
        self.assertEqual(extract_sql(""), "")

    def test_string_with_only_whitespace(self):
        self.assertEqual(extract_sql("   \n  "), "")

    def test_string_with_no_sql_just_text_and_backticks(self):
        text = "This is some text ```but not sql``` and more text."
        # According to current extract_sql logic, if "```sql" is not found,
        # it looks for "```". So this will extract "but not sql".
        self.assertEqual(extract_sql(text), "but not sql")

    def test_string_with_no_sql_and_no_backticks(self):
        text = "This is just plain text without SQL."
        self.assertEqual(extract_sql(text), "This is just plain text without SQL.")

    def test_multiple_sql_blocks_prefers_first_sql_tagged(self):
        text = "```\nSELECT 1;\n``` ... ```sql\nSELECT 2;\n``` ... ```\nSELECT 3;\n```"
        # Expects the first one with ```sql tag
        self.assertEqual(extract_sql(text), "SELECT 2;")

    def test_multiple_sql_blocks_no_sql_tag_prefers_first_generic(self):
        text = "```\nSELECT 1;\n``` ... ```\nSELECT 2;\n```"
        # Expects the first one with ``` tag if no ```sql is present
        self.assertEqual(extract_sql(text), "SELECT 1;")

    def test_sql_with_internal_backticks_in_comments_or_strings(self):
        # This tests if the regex is not too greedy or easily confused.
        # The current regex should handle this fine as it looks for the start/end ``` markers.
        sql_query = "SELECT `col` FROM test -- a comment with `backticks`"
        text_with_sql_tag = f"```sql\n{sql_query}\n```"
        self.assertEqual(extract_sql(text_with_sql_tag), sql_query)

        text_without_sql_tag = f"```\n{sql_query}\n```"
        self.assertEqual(extract_sql(text_without_sql_tag), sql_query)

        text_plain = sql_query
        self.assertEqual(extract_sql(text_plain), sql_query)


class TestFixSqlWithLlm(unittest.TestCase):

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.extract_sql') # Mock the extract_sql used by fix_sql_with_llm
    def test_fix_sql_successful_correction(self, mock_extract_sql, mock_llm_invoke):
        original_sql = "SELECT name FROM users WHER age > 30"
        error_message = "Syntax error near WHER"
        rag_tables = "Table users: name TEXT, age INTEGER"
        rag_queries = "SELECT name FROM users WHERE department = 'Sales'"
        user_query = "Show users older than 30"

        llm_corrected_sql_raw = "```sql\nSELECT name FROM users WHERE age > 30\n```"
        llm_corrected_sql_clean = "SELECT name FROM users WHERE age > 30"

        mock_llm_invoke.return_value = MagicMock(content=llm_corrected_sql_raw)
        mock_extract_sql.return_value = llm_corrected_sql_clean

        result = fix_sql_with_llm(original_sql, error_message, rag_tables, rag_queries, user_query)

        mock_llm_invoke.assert_called_once()
        prompt_arg = mock_llm_invoke.call_args[0][0]
        self.assertIn(original_sql, prompt_arg)
        self.assertIn(error_message, prompt_arg)
        self.assertIn(rag_tables, prompt_arg)
        self.assertIn(rag_queries, prompt_arg)
        self.assertIn(user_query, prompt_arg)

        mock_extract_sql.assert_called_once_with(llm_corrected_sql_raw)
        self.assertEqual(result, llm_corrected_sql_clean)

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.extract_sql')
    def test_fix_sql_llm_returns_faulty_sql(self, mock_extract_sql, mock_llm_invoke):
        original_sql = "SELECT name FROM users WHER age > 30"
        error_message = "Syntax error near WHER"
        # ... other args ...

        llm_still_faulty_sql_raw = "```SELECT name FROM users WHERE age > 30; -- Still not perfect```"
        llm_still_faulty_sql_clean = "SELECT name FROM users WHERRE age > 30; -- Still not perfect" # Example, extract_sql cleans it

        mock_llm_invoke.return_value = MagicMock(content=llm_still_faulty_sql_raw)
        mock_extract_sql.return_value = llm_still_faulty_sql_clean

        result = fix_sql_with_llm(original_sql, error_message, "tables", "queries", "userq")

        mock_extract_sql.assert_called_once_with(llm_still_faulty_sql_raw)
        self.assertEqual(result, llm_still_faulty_sql_clean)

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.extract_sql')
    def test_fix_sql_llm_returns_non_sql_content(self, mock_extract_sql, mock_llm_invoke):
        original_sql = "SELECT name FROM users WHER age > 30"
        error_message = "Syntax error near WHER"
        # ... other args ...

        llm_non_sql_response_raw = "I am unable to fix this SQL."
        # extract_sql would return this as is if no backticks
        llm_non_sql_response_clean = "I am unable to fix this SQL."

        mock_llm_invoke.return_value = MagicMock(content=llm_non_sql_response_raw)
        mock_extract_sql.return_value = llm_non_sql_response_clean

        result = fix_sql_with_llm(original_sql, error_message, "tables", "queries", "userq")

        mock_extract_sql.assert_called_once_with(llm_non_sql_response_raw)
        self.assertEqual(result, llm_non_sql_response_clean)


class TestTransformSqlError(unittest.TestCase):
    def test_no_such_table(self):
        raw_error = "no such table: my_table"
        expected = "指定されたテーブルが見つからなかったため、クエリを実行できませんでした。テーブル名を確認してください。(詳細: no such table: my_table)"
        self.assertEqual(transform_sql_error(raw_error), expected)

    def test_no_such_table_mixed_case(self):
        raw_error = "No SuCh TaBlE: SalesData"
        expected = "指定されたテーブルが見つからなかったため、クエリを実行できませんでした。テーブル名を確認してください。(詳細: No SuCh TaBlE: SalesData)"
        self.assertEqual(transform_sql_error(raw_error), expected)

    def test_no_such_column(self):
        raw_error = "no such column: my_column"
        expected = "指定された列が見つからなかったため、クエリを実行できませんでした。列名を確認してください。(詳細: no such column: my_column)"
        self.assertEqual(transform_sql_error(raw_error), expected)

    def test_no_such_column_with_table_hint(self):
        raw_error = "table users has no column named non_existent_field" # Example from other DBs, SQLite is usually simpler
        # Assuming current transform_sql_error only checks for "no such column" substring for this category.
        # If more sophisticated parsing is added later, this test might need adjustment.
        # For now, based on current logic, it should fall into generic if "no such column" is not present.
        # Let's test with an actual SQLite "no such column" variant.
        sqlite_like_error = "no such column: non_existent_field"
        expected = "指定された列が見つからなかったため、クエリを実行できませんでした。列名を確認してください。(詳細: no such column: non_existent_field)"
        self.assertEqual(transform_sql_error(sqlite_like_error), expected)


    def test_syntax_error(self):
        raw_error = 'near "SELECTX": syntax error'
        expected = 'SQL構文にエラーがあります。クエリを確認してください。(詳細: near "SELECTX": syntax error)'
        self.assertEqual(transform_sql_error(raw_error), expected)

    def test_syntax_error_various_messages(self):
        raw_error = "incomplete input" # another syntax error example
        expected = 'SQL構文にエラーがあります。クエリを確認してください。(詳細: incomplete input)'
        self.assertEqual(transform_sql_error(raw_error), expected)

    def test_uncommon_generic_error(self):
        raw_error = "database disk image is malformed"
        expected = "SQLクエリの処理中に予期せぬエラーが発生しました。管理者に連絡してください。(詳細: database disk image is malformed)"
        self.assertEqual(transform_sql_error(raw_error), expected)

    def test_empty_string_input(self):
        raw_error = ""
        # Current implementation would treat empty string as a generic error.
        expected = "SQLクエリの処理中に予期せぬエラーが発生しました。管理者に連絡してください。(詳細: )"
        self.assertEqual(transform_sql_error(raw_error), expected)

    def test_none_input(self):
        # This will cause an AttributeError because the function expects a string for .lower()
        with self.assertRaises(AttributeError):
            transform_sql_error(None)
        # If we want to handle None gracefully, the function needs a check.
        # For now, testing current behavior.


class TestInterpretNode(unittest.TestCase):

    @patch('files.backend_codes.llm')
    def test_interpret_node_valid_data(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="This is a great interpretation of the data.")
        sample_data = collections.OrderedDict({
            "Sales Data": [{"item": "A", "revenue": 100}, {"item": "B", "revenue": 150}],
            "Inventory Data": [{"item": "A", "stock": 20}, {"item": "B", "stock": 30}]
        })
        state = MyState(latest_df=sample_data)

        result_state = interpret_node(state)

        mock_llm.invoke.assert_called_once()
        prompt_arg = mock_llm.invoke.call_args[0][0] # Gets the list of messages/string passed to invoke

        # Verify the prompt more thoroughly for each key-value pair in the sample_data
        for req, data_list in sample_data.items():
            self.assertIn(req, str(prompt_arg))
            if data_list: # If data_list is not empty
                for row in data_list:
                    for key, value in row.items():
                        self.assertIn(str(key), str(prompt_arg))
                        self.assertIn(str(value), str(prompt_arg))
            else:
                # This case is handled by test_interpret_node_empty_data_list_in_dict
                pass

        self.assertEqual(result_state["interpretation"], "This is a great interpretation of the data.")
        self.assertEqual(result_state["condition"], "解釈完了")

    @patch('files.backend_codes.llm')
    def test_interpret_node_empty_data_dict(self, mock_llm):
        state = MyState(latest_df=collections.OrderedDict())
        result_state = interpret_node(state)
        self.assertEqual(result_state["interpretation"], "データが取得されましたが、内容は空です。")
        self.assertEqual(result_state["condition"], "解釈失敗")
        mock_llm.invoke.assert_not_called()

    @patch('files.backend_codes.llm')
    def test_interpret_node_empty_data_list_in_dict(self, mock_llm):
        # This test had a slight duplication in calling interpret_node(state) twice. Corrected.
        mock_llm.invoke.return_value = MagicMock(content="No data was available for Empty Sales.")
        state = MyState(latest_df=collections.OrderedDict({"Empty Sales": []}))

        result_state = interpret_node(state) # Call interpret_node once

        mock_llm.invoke.assert_called_once()
        prompt_arg = mock_llm.invoke.call_args[0][0]
        self.assertIn("Empty Sales", str(prompt_arg))
        self.assertIn("(この要件に対するデータはありません)", str(prompt_arg))
        self.assertEqual(result_state["interpretation"], "No data was available for Empty Sales.")
        self.assertEqual(result_state["condition"], "解釈完了")


    @patch('files.backend_codes.llm')
    def test_interpret_node_none_data(self, mock_llm):
        state = MyState(latest_df=None)
        result_state = interpret_node(state)
        self.assertEqual(result_state["interpretation"], "まだデータがありません。先にSQL質問をするか、メタデータ検索を試してください。")
        self.assertEqual(result_state["condition"], "解釈失敗")
        mock_llm.invoke.assert_not_called()

    @patch('files.backend_codes.llm')
    def test_interpret_node_llm_empty_string_response(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="") # LLM returns empty string
        sample_data = collections.OrderedDict({
            "Sales Data": [{"item": "A", "revenue": 100}]
        })
        state = MyState(latest_df=sample_data)

        result_state = interpret_node(state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["interpretation"], "") # Interpretation should be empty
        self.assertEqual(result_state["condition"], "解釈完了") # Still considered a successful interpretation

    @patch('files.backend_codes.llm')
    def test_interpret_node_llm_none_content_response(self, mock_llm):
        # Simulate LLM response where content attribute is None
        mock_response = MagicMock()
        mock_response.content = None
        mock_llm.invoke.return_value = mock_response

        sample_data = collections.OrderedDict({
            "Sales Data": [{"item": "A", "revenue": 100}]
        })
        state = MyState(latest_df=sample_data)

        # Expect AttributeError because interpret_node tries to call .strip() on None
        with self.assertRaises(AttributeError):
            interpret_node(state)

        mock_llm.invoke.assert_called_once()
        # Depending on exact behavior, condition might not be set or might be an error state.
        # Since the function errors out before setting these, we don't check them here.

class TestChartNode(unittest.TestCase):

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'chart_bytes')
    @patch('files.backend_codes.base64.b64encode')
    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.PythonAstREPLTool') # Mock the tool itself
    @patch('files.backend_codes.initialize_agent') # Mock agent initialization
    def test_chart_node_successful_generation(
        self, mock_initialize_agent, mock_python_tool, mock_os_path_exists, mock_b64encode, mock_open_file
    ):
        # Setup Mocks
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke.return_value = {"output": "Agent created output.png"} # Simulate agent execution

        mock_python_tool_instance = MagicMock()
        mock_python_tool.return_value = mock_python_tool_instance

        mock_os_path_exists.return_value = True # Simulate output.png exists
        mock_b64encode.return_value = "encoded_chart_data"

        # Use the first entry from latest_df for charting
        chart_requirement_name = "Sales"
        chart_data_list = [{"category": "A", "amount": 100}, {"category": "B", "amount": 150}]
        sample_data_ordered_dict = collections.OrderedDict({
            chart_requirement_name: chart_data_list,
            "Other Data": [{"info": "test"}] # Add another entry to ensure only the first is used
        })
        user_input = "Plot sales data"
        state = MyState(latest_df=sample_data_ordered_dict, input=user_input)

        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()

        # Check that PythonAstREPLTool was instantiated with the correct DataFrame
        tool_args = mock_python_tool.call_args
        self.assertIn('df', tool_args.kwargs['locals'])
        passed_df_to_tool = tool_args.kwargs['locals']['df']
        self.assertIsInstance(passed_df_to_tool, pd.DataFrame)
        # Expected DataFrame should be created from the first item's data in the OrderedDict
        expected_df_for_tool = pd.DataFrame(chart_data_list)
        pd.testing.assert_frame_equal(passed_df_to_tool, expected_df_for_tool)

        mock_agent_instance.invoke.assert_called_once()
        agent_prompt = mock_agent_instance.invoke.call_args[0][0]

        self.assertIn("最適なグラフをsns(seaborn)で作成して", agent_prompt)
        self.assertIn("sns.set(font='IPAexGothic')", agent_prompt)
        self.assertIn(user_input, agent_prompt) # User input
        self.assertIn(chart_requirement_name, agent_prompt) # Requirement name (key from latest_df)
        self.assertIn(expected_df_for_tool.head().to_string(), agent_prompt) # String representation of DataFrame head

        mock_os_path_exists.assert_called_once_with("output.png")
        mock_b64encode.assert_called_once_with(b"chart_bytes")
        mock_open_file.assert_called_once_with("output.png", "rb")

        self.assertEqual(result_state["chart_result"], "encoded_chart_data")
        self.assertEqual(result_state["condition"], "グラフ化完了")

    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_empty_latest_df(self, mock_initialize_agent):
        state = MyState(latest_df=collections.OrderedDict(), input="Plot this")
        result_state = chart_node(state)
        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "グラフ化失敗")
        mock_initialize_agent.assert_not_called()

    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_none_latest_df(self, mock_initialize_agent):
        state = MyState(latest_df=None, input="Plot this")
        result_state = chart_node(state)
        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "グラフ化失敗")
        mock_initialize_agent.assert_not_called()

    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.initialize_agent') # To ensure it's called
    @patch('files.backend_codes.PythonAstREPLTool')
    def test_chart_node_output_file_not_created(
        self, mock_python_tool, mock_initialize_agent, mock_os_path_exists
    ):
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke.return_value = {"output": "Agent ran but did not create file"}

        mock_os_path_exists.return_value = False # output.png does not exist

        sample_data = collections.OrderedDict({"Sales": [{"category": "A", "amount": 100}]})
        state = MyState(latest_df=sample_data, input="Plot sales data")

        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()
        mock_agent_instance.invoke.assert_called_once()
        mock_os_path_exists.assert_called_once_with("output.png")
        # Ensure PythonAstREPLTool was still called
        mock_python_tool.assert_called_once()


        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "グラフ化失敗")

    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_agent_invalid_code(
        self, mock_initialize_agent, mock_python_tool, mock_os_path_exists
    ):
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        # Simulate agent's Python code failing (e.g. PythonAstREPLTool's execution fails)
        # The current chart_node wraps the agent call in a try-except Exception.
        mock_agent_instance.invoke.side_effect = Exception("Invalid Python code generated by agent")

        sample_data = collections.OrderedDict({"Sales": [{"category": "A", "amount": 100}]})
        state = MyState(latest_df=sample_data, input="Plot sales data for test_chart_node_agent_invalid_code")

        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()
        mock_python_tool.assert_called_once() # Tool is prepared
        mock_agent_instance.invoke.assert_called_once() # Agent is invoked

        # os.path.exists might not be called if invoke fails and is caught.
        # If the exception happens during agent.invoke, the flow might go to the except block
        # before os.path.exists("output.png") is called.
        # Let's verify it's not called, or if it is, it's with "output.png" and returns False.
        # Given the try-catch in chart_node, it's likely not called after error.
        # However, the code *tries* to delete "output.png" in the except block, so os.path.exists might be called there.
        # For robustness, we can allow it to be called.
        if mock_os_path_exists.called:
            mock_os_path_exists.assert_any_call("output.png")


        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "グラフ化失敗")
        self.assertIn("Failed to generate chart due to error: Invalid Python code generated by agent", result_state.get("error", ""))


    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'chart_bytes_various')
    @patch('files.backend_codes.base64.b64encode')
    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_various_data_types(
        self, mock_initialize_agent, mock_python_tool, mock_os_path_exists, mock_b64encode, mock_open_file
    ):
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke.return_value = {"output": "Agent created output.png"}

        mock_os_path_exists.return_value = True # Simulate output.png exists
        mock_b64encode.return_value = "encoded_chart_various_data"

        chart_requirement_name = "DiverseData"
        # Data with numerical, categorical, and date-like strings
        diverse_data_list = [
            {"id": 1, "product_category": "Electronics", "sales_date": "2023-01-15", "units_sold": 10, "revenue": 1200.50},
            {"id": 2, "product_category": "Books", "sales_date": "2023-01-16", "units_sold": 50, "revenue": 750.00},
            {"id": 3, "product_category": "Electronics", "sales_date": "2023-01-17", "units_sold": 8, "revenue": 980.75},
            {"id": 4, "product_category": "Home Goods", "sales_date": "2023-01-18", "units_sold": 25, "revenue": 300.25},
        ]
        sample_data_ordered_dict = collections.OrderedDict({chart_requirement_name: diverse_data_list})
        user_input = "Plot diverse data types"
        state = MyState(latest_df=sample_data_ordered_dict, input=user_input)

        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()

        tool_args = mock_python_tool.call_args
        self.assertIn('df', tool_args.kwargs['locals'])
        passed_df_to_tool = tool_args.kwargs['locals']['df']
        self.assertIsInstance(passed_df_to_tool, pd.DataFrame)
        expected_df_for_tool = pd.DataFrame(diverse_data_list)
        pd.testing.assert_frame_equal(passed_df_to_tool, expected_df_for_tool)

        mock_agent_instance.invoke.assert_called_once()
        agent_prompt = mock_agent_instance.invoke.call_args[0][0]

        self.assertIn("最適なグラフをsns(seaborn)で作成して", agent_prompt)
        self.assertIn("sns.set(font='IPAexGothic')", agent_prompt)
        self.assertIn(user_input, agent_prompt)
        self.assertIn(chart_requirement_name, agent_prompt)
        self.assertIn(expected_df_for_tool.head().to_string(), agent_prompt) # Key check for diverse data

        mock_os_path_exists.assert_called_once_with("output.png")
        mock_b64encode.assert_called_once_with(b"chart_bytes_various")
        mock_open_file.assert_called_once_with("output.png", "rb")

        self.assertEqual(result_state["chart_result"], "encoded_chart_various_data")
        self.assertEqual(result_state["condition"], "グラフ化完了")


class TestMetadataRetrievalNode(unittest.TestCase):

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.llm')
    def test_metadata_retrieval_successful(self, mock_llm, mock_vs_tables_search):
        user_query = "What are the columns in sales_table?"
        mock_rag_docs = [
            MagicMock(page_content="Table: sales_table Columns: id, date, amount"),
            MagicMock(page_content="Table: product_table Columns: id, name, price")
        ]
        mock_vs_tables_search.return_value = mock_rag_docs

        expected_llm_answer = "The sales_table has columns: id, date, and amount."
        mock_llm.invoke.return_value = MagicMock(content=expected_llm_answer)

        state = MyState(input=user_query)
        result_state = metadata_retrieval_node(state)

        mock_vs_tables_search.assert_called_once_with(user_query, k=3)

        mock_llm.invoke.assert_called_once()
        llm_prompt_arg_str = str(mock_llm.invoke.call_args[0][0]) # Get the prompt string

        self.assertIn(user_query, llm_prompt_arg_str)
        # Check for the fixed instruction part of the prompt
        self.assertIn("以下のテーブル定義情報を参照して、ユーザーの質問に答えてください。", llm_prompt_arg_str)
        # Check for each piece of RAG content
        for doc in mock_rag_docs:
            self.assertIn(doc.page_content, llm_prompt_arg_str)

        self.assertEqual(result_state["metadata_answer"], expected_llm_answer)
        self.assertEqual(result_state["condition"], "メタデータ検索完了")

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.llm')
    def test_metadata_retrieval_no_rag_docs_found(self, mock_llm, mock_vs_tables_search):
        user_query = "Tell me about a non_existent_table."
        mock_vs_tables_search.return_value = [] # No documents found

        expected_llm_answer = "関連する情報が見つかりませんでした。"
        mock_llm.invoke.return_value = MagicMock(content=expected_llm_answer)

        state = MyState(input=user_query)
        result_state = metadata_retrieval_node(state)

        mock_vs_tables_search.assert_called_once_with(user_query, k=3)

        mock_llm.invoke.assert_called_once()
        llm_prompt_arg_str = str(mock_llm.invoke.call_args[0][0])

        self.assertIn(user_query, llm_prompt_arg_str)
        self.assertIn("以下のテーブル定義情報を参照して、ユーザーの質問に答えてください。", llm_prompt_arg_str)
        # The RAG context part should be minimal or indicate no info
        # For example, if context is "\n".join(docs), it would be empty.
        # The exact check depends on how an empty doc list is formatted in the prompt.
        # Assuming it results in an empty string or just newlines for the context part.
        # A simple check is that specific table info (from previous test) is NOT there.
        self.assertNotIn("Table: sales_table", llm_prompt_arg_str)


        self.assertEqual(result_state["metadata_answer"], expected_llm_answer)
        self.assertEqual(result_state["condition"], "メタデータ検索完了")

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.llm')
    def test_metadata_retrieval_llm_generic_response(self, mock_llm, mock_vs_tables_search):
        user_query = "What are the columns in yet_another_table?"
        mock_rag_docs = [
            MagicMock(page_content="Table: yet_another_table Columns: colA, colB")
        ]
        mock_vs_tables_search.return_value = mock_rag_docs

        # Test with empty string response
        generic_llm_answer_empty = ""
        mock_llm.invoke.return_value = MagicMock(content=generic_llm_answer_empty)

        state_empty_resp = MyState(input=user_query)
        result_state_empty = metadata_retrieval_node(state_empty_resp)

        mock_vs_tables_search.assert_called_with(user_query, k=3) # Called again
        mock_llm.invoke.assert_called_with(unittest.mock.ANY) # Called again

        llm_prompt_arg_str_empty = str(mock_llm.invoke.call_args[0][0])
        self.assertIn(user_query, llm_prompt_arg_str_empty)
        self.assertIn("Table: yet_another_table Columns: colA, colB", llm_prompt_arg_str_empty)
        self.assertIn("以下のテーブル定義情報を参照して、ユーザーの質問に答えてください。", llm_prompt_arg_str_empty)

        self.assertEqual(result_state_empty["metadata_answer"], generic_llm_answer_empty)
        self.assertEqual(result_state_empty["condition"], "メタデータ検索完了")

        # Test with a generic non-helpful response
        generic_llm_answer_non_helpful = "Could not process the request based on the provided information."
        mock_llm.invoke.return_value = MagicMock(content=generic_llm_answer_non_helpful)

        state_generic_resp = MyState(input=user_query) # Fresh state for this call
        result_state_generic = metadata_retrieval_node(state_generic_resp)

        mock_vs_tables_search.assert_called_with(user_query, k=3) # Called again
        mock_llm.invoke.assert_called_with(unittest.mock.ANY) # Called again

        self.assertEqual(result_state_generic["metadata_answer"], generic_llm_answer_non_helpful)
        self.assertEqual(result_state_generic["condition"], "メタデータ検索完了")


class TestClearDataNode(unittest.TestCase):
    def test_clear_data_node_resets_state_correctly(self):
        initial_state = MyState(
            input="リセットして",
            intent_list=["clear_data_intent"],
            latest_df=collections.OrderedDict({"some_data": [{"col": "val"}]}),
            df_history=[{"id": "old_id", "query": "old_query", "dataframe_dict": [{"data": "content"}], "SQL": "OLD SQL"}],
            SQL="SELECT * FROM old_table",
            interpretation="Old interpretation",
            chart_result="old_chart_base64",
            metadata_answer="Old metadata value", # Explicitly set for preservation check
            condition="initial_condition", # This will be overwritten
            error="initial_error", # This will be cleared
            query_history=["q1", "q2", "q3"] # Explicitly set for clearing check
        )

        cleared_state = clear_data_node(initial_state)

        # Assert preserved fields
        self.assertEqual(cleared_state["input"], "リセットして")
        self.assertEqual(cleared_state["intent_list"], ["clear_data_intent"])
        self.assertEqual(cleared_state["metadata_answer"], "Old metadata value")

        # Assert cleared fields
        self.assertEqual(cleared_state["latest_df"], collections.OrderedDict())
        self.assertEqual(cleared_state["df_history"], [])
        self.assertIsNone(cleared_state["SQL"])
        self.assertEqual(cleared_state["interpretation"], DATA_CLEARED_MESSAGE)
        self.assertIsNone(cleared_state["chart_result"])
        self.assertIsNone(cleared_state["error"])
        self.assertEqual(cleared_state["query_history"], [])

        # Assert condition
        self.assertEqual(cleared_state["condition"], "データクリア完了")

    def test_clear_data_node_minimal_initial_state(self):
        initial_state_minimal = MyState(
            input="リセット",
            intent_list=[], # Test with empty list
            latest_df=None, # Test with None
            df_history=None, # Test with None
            SQL=None,
            interpretation=None,
            chart_result=None,
            metadata_answer=None, # Test preservation of None
            condition="some_condition", # This will be overwritten
            error=None,
            query_history=None # Test with None
        )

        cleared_state = clear_data_node(initial_state_minimal)

        # Assert preserved fields
        self.assertEqual(cleared_state["input"], "リセット")
        self.assertEqual(cleared_state["intent_list"], []) # Preserved as empty
        self.assertIsNone(cleared_state["metadata_answer"]) # Preserved as None

        # Assert cleared fields (should be set to their "cleared" defaults)
        self.assertEqual(cleared_state["latest_df"], collections.OrderedDict())
        self.assertEqual(cleared_state["df_history"], [])
        self.assertIsNone(cleared_state["SQL"])
        self.assertEqual(cleared_state["interpretation"], DATA_CLEARED_MESSAGE)
        self.assertIsNone(cleared_state["chart_result"])
        self.assertIsNone(cleared_state["error"])
        self.assertEqual(cleared_state["query_history"], [])

        # Assert condition
        self.assertEqual(cleared_state["condition"], "データクリア完了")


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        # Build a new workflow for each test to ensure isolation
        self.workflow = build_workflow()

    @patch('files.backend_codes.llm') # Mock for classify and extract_data_requirements, interpret
    @patch('files.backend_codes.vectorstore_tables.similarity_search') # Mock for RAG in sql_node (if reached)
    @patch('files.backend_codes.vectorstore_queries.similarity_search') # Mock for RAG in sql_node (if reached)
    @patch('files.backend_codes.try_sql_execute') # Mock for DB in sql_node (if reached)
    @patch('files.backend_codes.PythonAstREPLTool') # Mock for chart_node's tool
    def test_workflow_all_data_found_then_interpret(
        self, mock_repl_tool, mock_try_sql, mock_vs_queries, mock_vs_tables, mock_llm
    ):
        user_input = "A商品の売上と顧客属性について教えて"
        config = {"configurable": {"thread_id": "test_thread_all_found"}}

        # --- Mocking sequence for LLM calls ---
        # 1. classify_intent_node
        mock_llm.invoke.side_effect = [
            MagicMock(content="データ取得,データ解釈"),  # classify_intent_node result
            MagicMock(content="A商品の売上,顧客属性"), # extract_data_requirements_node result
            MagicMock(content="Interpreted results about A sales and customer attributes.") # interpret_node result
        ]

        # --- Initial State (simulating history) ---
        # We need to manually set up the df_history part of the state for find_similar_query
        # This is a bit tricky with the actual checkpointer.
        # A cleaner way for workflow tests is often to pre-populate the checkpointer's memory
        # or mock the checkpointer's get/put methods.
        # For now, let's assume we can pass a pre-populated df_history.
        # However, the graph starts fresh. find_similar_query will use its passed state.
        # So, we need to ensure that when find_similar_query_node is called,
        # its 'df_history' is populated.
        # The current workflow structure doesn't allow easily injecting df_history for a specific run via invoke.
        # Let's patch find_similar_query_node itself for this specific workflow test to control its output,
        # or accept that it will initially find nothing if history is empty.

        # For this test, let's assume find_similar_query_node works as tested unit-wise,
        # and we want to test the workflow path. So, we'll make it so "all_data_found" is the condition.
        # To do this, we can patch `find_similar_query_node` to return a specific state.

        predefined_data_for_A = [{"product": "A", "sales": 120}]
        predefined_data_for_cust = [{"attr": "loyal"}]

        # Patching the node itself to control its output for the workflow test
        # This is an alternative to deeply populating history for a workflow test run
        with patch('files.backend_codes.find_similar_query_node') as mock_find_similar:
            mock_find_similar.return_value = {
                "input": user_input,
                "intent_list": ["データ取得", "データ解釈"],
                "data_requirements": ["A商品の売上", "顧客属性"],
                "latest_df": collections.OrderedDict([
                    ("A商品の売上", predefined_data_for_A),
                    ("顧客属性", predefined_data_for_cust)
                ]),
                "missing_data_requirements": [],
                "condition": "all_data_found", # CRITICAL for routing
                "df_history": [ # Dummy history that would lead to this state
                     {"id": "h1", "query": "A商品の売上", "dataframe_dict": predefined_data_for_A, "SQL": "S1"},
                     {"id": "h2", "query": "顧客属性", "dataframe_dict": predefined_data_for_cust, "SQL": "S2"},
                ],
                "query_history": [user_input]
            }

            # Invoke the workflow
            final_state = self.workflow.invoke(
                {"input": user_input, "df_history": []}, # df_history here is for the very start, find_similar is patched
                config=config
            )

        # --- Assertions ---
        # 1. classify_intent_node (mocked)
        # 2. extract_data_requirements_node (mocked)
        # 3. find_similar_query_node (patched to return "all_data_found")
        # 4. Next should be interpret_node (since "データ解釈" is in intent_list and "グラフ作成" is not)

        # Check final state from interpret_node
        self.assertEqual(final_state["interpretation"], "Interpreted results about A sales and customer attributes.")
        self.assertEqual(final_state["latest_df"]["A商品の売上"], predefined_data_for_A) # Ensure latest_df propagated
        self.assertEqual(final_state["condition"], "解釈完了") # Condition from interpret_node

        # Check LLM calls
        # Call 1 (classify): "データ取得,データ解釈"
        # Call 2 (extract_data_req): "A商品の売上,顧客属性"
        # Call 3 (interpret): "Interpreted results..."
        self.assertEqual(mock_llm.invoke.call_count, 3)

        llm_calls = mock_llm.invoke.call_args_list
        # Call 1: Classify intent
        self.assertIn("ユーザーの質問の意図を判定してください。", str(llm_calls[0].args[0])) # Prompt for classify_intent
        self.assertIn(user_input, str(llm_calls[0].args[0]))
        # Call 2: Extract data requirements
        self.assertIn("必要となる具体的なデータの要件を抽出してください。", str(llm_calls[1].args[0])) # Prompt for extract_data_requirements
        self.assertIn(user_input, str(llm_calls[1].args[0]))
        # Call 3: Interpret node
        self.assertIn("以下のSQLクエリ実行結果群について", str(llm_calls[2].args[0])) # Prompt for interpret_node
        self.assertIn("A商品の売上", str(llm_calls[2].args[0])) # Check if data is in prompt
        self.assertIn("顧客属性", str(llm_calls[2].args[0]))

        # Ensure mocks for nodes not reached (sql_node, chart_node) were not called
        mock_vs_tables.assert_not_called()
        mock_vs_queries.assert_not_called()
        mock_try_sql.assert_not_called()
        mock_repl_tool.assert_not_called()

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    # Removed try_sql_execute from direct patches here as it's used by sql_node, which is implicitly tested.
    # Same for PythonAstREPLTool, uuid.
    def test_workflow_metadata_retrieval_path(
        self, mock_vs_queries, mock_vs_tables, mock_llm # Mocks are applied from bottom up
    ):
        user_input = "sales_dataテーブルにはどんなカラムがありますか？"
        config = {"configurable": {"thread_id": "test_thread_metadata"}}

        # --- Mocking sequence for LLM calls ---
        # 1. classify_intent_node: returns "メタデータ検索"
        # 2. metadata_retrieval_node: returns the metadata answer
        mock_llm.invoke.side_effect = [
            MagicMock(content="メタデータ検索"),
            MagicMock(content="sales_dataテーブルには、'date', 'product_id', 'sales_amount' カラムがあります。")
        ]

        # Mock vectorstore_tables used by metadata_retrieval_node
        mock_vs_tables.return_value = [
            MagicMock(page_content="Table: sales_data Columns: date, product_id, sales_amount"),
            MagicMock(page_content="Table: customers Columns: customer_id, name, segment")
        ]

        # Mocks that should NOT be called for this path
        # These are good candidates to move into specific test setups if they are not shared,
        # or ensure they are reset if tests share them.
        # For this test, we are primarily checking the LLM and vectorstore calls.
        # Other nodes like extract_data_requirements, find_similar_query, sql, chart, interpret
        # should not be triggered. We can verify this by checking the LLM call count and specific prompts.

        # Invoke the workflow
        final_state = self.workflow.invoke({"input": user_input}, config=config)

        # --- Assertions ---
        # Path: classify -> metadata_retrieval -> END
        self.assertEqual(final_state["condition"], "メタデータ検索完了")
        self.assertEqual(final_state["metadata_answer"], "sales_dataテーブルには、'date', 'product_id', 'sales_amount' カラムがあります。")

        # Check LLM calls
        self.assertEqual(mock_llm.invoke.call_count, 2)
        llm_calls = mock_llm.invoke.call_args_list

        # Call 1: Classify intent
        self.assertIn("ユーザーの質問の意図を判定してください。", str(llm_calls[0].args[0]))
        self.assertIn(user_input, str(llm_calls[0].args[0]))

        # Call 2: Metadata retrieval node
        self.assertIn("以下のテーブル定義情報を参照して、ユーザーの質問に答えてください。", str(llm_calls[1].args[0]))
        self.assertIn(user_input, str(llm_calls[1].args[0]))
        self.assertIn("Table: sales_data Columns: date, product_id, sales_amount", str(llm_calls[1].args[0])) # Check RAG content in prompt

        # Check vectorstore_tables call
        mock_vs_tables.assert_called_once_with(user_input, k=3)

        # Ensure other vectorstore (for queries in sql_node) was not called
        mock_vs_queries.assert_not_called()

        # By checking the LLM call count (2) and the specific prompts, we implicitly assert
        # that nodes like extract_data_requirements, sql_node (SQL generation part), interpret_node, chart_node
        # were not called as they would have made additional LLM calls with different prompts.

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.try_sql_execute') # Keep for sql_node
    @patch('files.backend_codes.PythonAstREPLTool') # For chart_node
    @patch('files.backend_codes.uuid.uuid4') # For sql_node history
    def test_workflow_missing_data_then_sql_then_chart(
        self, mock_uuid, mock_repl_tool, mock_try_sql, mock_vs_queries, mock_vs_tables, mock_llm
    ):
        user_input = "A商品の売上とB商品の在庫をグラフにして"
        config = {"configurable": {"thread_id": "test_thread_missing_sql_chart"}}
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="wf_sql_".ljust(8, '0')))

        # --- Mocking sequence for LLM calls ---
        # 1. classify_intent_node: "データ取得,グラフ作成"
        # 2. extract_data_requirements_node: "A商品の売上,B商品の在庫"
        # 3. sql_node (for B商品の在庫): generates SQL "SELECT * FROM B_inventory;"
        # 4. chart_node: Python code for charting
        mock_llm.invoke.side_effect = [
            MagicMock(content="データ取得,グラフ作成"),           # Call 1: classify_intent
            MagicMock(content="A商品の売上,B商品の在庫"),       # Call 2: extract_data_requirements
            MagicMock(content="SELECT * FROM B_inventory;"), # Call 3: sql_node (SQL gen for B商品の在庫)
            MagicMock(content="print('Chart generated')")    # Call 4: chart_node (agent prompt)
        ]

        # --- Data for find_similar_query_node ---
        # Assume "A商品の売上" is found in history, "B商品の在庫" is missing.
        data_A_sales = [{"product": "A", "sales": 200}]
        history_entry_A = {"id": "h_A", "query": "A商品の売上", "dataframe_dict": data_A_sales, "SQL": "SELECT A"}

        # --- Data for sql_node (fetching B商品の在庫) ---
        data_B_inventory_df = pd.DataFrame([{"item": "B", "stock": 30}])

        # Mock RAG for sql_node (only for B商品の在庫)
        mock_vs_tables.return_value = [MagicMock(page_content="Table info for B")]
        mock_vs_queries.return_value = [MagicMock(page_content="Query info for B")]
        # Mock DB for sql_node
        mock_try_sql.return_value = (data_B_inventory_df, None)

        # Mock the chart tool execution
        mock_repl_tool_instance = mock_repl_tool.return_value
        # The agent's invoke method is what's called by the chart_node's agent.invoke(chart_prompt)
        # This should return a dict with an "output" key.
        mock_repl_tool_instance.invoke.return_value = {"output": "Chart tool executed, output.png created"}

        # Mock os.path.exists for chart_node to simulate chart output file
        with patch('files.backend_codes.os.path.exists') as mock_path_exists, \
             patch('files.backend_codes.base64.b64encode') as mock_b64encode, \
             patch('builtins.open', unittest.mock.mock_open(read_data=b"chart_bytes")) as mock_open_file, \
             patch('files.backend_codes.find_similar_query_node') as mock_find_similar: # Patch to control its output

            mock_path_exists.return_value = True # Simulate output.png exists
            mock_b64encode.return_value = "encoded_chart_data"

            # Configure mock_find_similar to return "missing_data"
            mock_find_similar.return_value = {
                "input": user_input,
                "intent_list": ["データ取得", "グラフ作成"],
                "data_requirements": ["A商品の売上", "B商品の在庫"],
                "latest_df": collections.OrderedDict([("A商品の売上", data_A_sales)]), # A is found
                "missing_data_requirements": ["B商品の在庫"], # B is missing
                "condition": "missing_data", # CRITICAL for routing to sql_node
                "df_history": [history_entry_A],
                "query_history": [user_input]
            }

            # Invoke the workflow
            # Initial state for the workflow can include pre-existing df_history if desired
            initial_workflow_state = {
                "input": user_input,
                "df_history": [history_entry_A], # History that find_similar_query_node should use
                 # Other fields like latest_df, SQL, etc., should be empty or None initially for this test
                "latest_df": collections.OrderedDict(),
                "SQL": None,
                "interpretation": None,
                "chart_result": None,
                "metadata_answer": None,
                "error": None,
                "query_history": [], # Will be populated by classify_intent_node
                "data_requirements": [], # Will be populated by extract_data_requirements_node
                "missing_data_requirements": [], # Will be populated by find_similar_query_node
            }
            final_state = self.workflow.invoke(initial_workflow_state, config=config)


        # --- Assertions ---
        # Path: classify -> extract_data -> find_similar (missing) -> sql -> chart -> END
        self.assertEqual(final_state["condition"], "グラフ化完了")
        self.assertEqual(final_state["chart_result"], "encoded_chart_data")

        # Check latest_df has both A (from history via find_similar) and B (from sql_node)
        self.assertIn("A商品の売上", final_state["latest_df"])
        self.assertEqual(final_state["latest_df"]["A商品の売上"], data_A_sales)
        self.assertIn("B商品の在庫", final_state["latest_df"])
        self.assertEqual(final_state["latest_df"]["B商品の在庫"], data_B_inventory_df.to_dict(orient="records"))

        # Check history updates from sql_node (one original, one new from sql_node)
        self.assertEqual(len(final_state["df_history"]), 2)
        self.assertTrue(any(h["query"] == "B商品の在庫" for h in final_state["df_history"]))

        # LLM Calls: classify, extract_req, sql_gen_B, chart_agent_prompt
        self.assertEqual(mock_llm.invoke.call_count, 4)
        llm_calls = mock_llm.invoke.call_args_list
        # Call 1: Classify intent
        self.assertIn("ユーザーの質問の意図を判定してください。", str(llm_calls[0].args[0]))
        self.assertIn(user_input, str(llm_calls[0].args[0]))
        # Call 2: Extract data requirements
        self.assertIn("必要となる具体的なデータの要件を抽出してください。", str(llm_calls[1].args[0]))
        self.assertIn(user_input, str(llm_calls[1].args[0]))
        # Call 3: SQL node (for B商品の在庫)
        self.assertIn("【現在の具体的なデータ取得要件】", str(llm_calls[2].args[0])) # Prompt for sql_node
        self.assertIn("B商品の在庫", str(llm_calls[2].args[0]))
        # Call 4: Chart node (agent prompt)
        # The chart_node's agent is initialized with PythonAstREPLTool, and agent.invoke is called.
        # The prompt to this agent.invoke is what we check here.
        agent_chart_prompt = final_state.get("_chart_agent_prompt_for_test", "") # Assuming we store it for test
                                                                                # Or, get from mock_repl_tool_instance.invoke.call_args

        # The PythonAstREPLTool's .invoke method is called by the agent framework.
        # The mock_repl_tool is the class, mock_repl_tool_instance is the instance.
        # The agent's invoke (which is different from the tool's invoke) gets the chart_prompt.
        # The PythonAstREPLTool itself is not directly invoked with the chart_prompt.
        # The chart_prompt is for the LLM part of the agent.
        # So, checking mock_llm.invoke.call_args_list[3] is correct for the agent's LLM call.
        self.assertIn("最適なグラフをsns(seaborn)で作成して", str(llm_calls[3].args[0]))
        self.assertIn(user_input, str(llm_calls[3].args[0]))
        self.assertIn("A商品の売上", str(llm_calls[3].args[0]))


        # RAG calls for B商品の在庫 in sql_node
        mock_vs_tables.assert_called_once_with("B商品の在庫", k=3)
        mock_vs_queries.assert_called_once_with("B商品の在庫", k=3)

        # DB call for B商品の在庫 in sql_node
        mock_try_sql.assert_called_once_with("SELECT * FROM B_inventory;")

        # The agent.invoke is called with the chart_prompt.
        # The PythonAstREPLTool instance's invoke method is called by the agent with the generated Python code.
        # This was mocked as: mock_repl_tool_instance.invoke.return_value = {"output": "Chart tool executed, output.png created"}
        mock_repl_tool_instance.invoke.assert_called_once() # This confirms the tool (Python code execution) was called.

        mock_path_exists.assert_called_once_with("output.png")
        mock_b64encode.assert_called_once_with(b"chart_bytes")


    @patch('files.backend_codes.llm')
    # No other mocks needed as clear_data_node doesn't use them.
    def test_workflow_clear_data_command(self, mock_llm):
        user_input = "SYSTEM_CLEAR_HISTORY" # Special command
        config = {"configurable": {"thread_id": "test_thread_clear_data"}}

        # Initial state with some data to ensure it gets cleared
        initial_state_for_clear = {
            "input": user_input,
            "latest_df": collections.OrderedDict({"some_data": [{"col": "val"}]}),
            "df_history": [{"id": "old_id", "query": "old_q", "dataframe_dict": [], "SQL": "OLD SQL"}],
            "SQL": "SELECT * FROM old_table",
            "interpretation": "Old interpretation",
            "chart_result": "old_chart_base64",
            "metadata_answer": "Old metadata to be preserved",
            "error": "some_error",
            "query_history": ["q1", "q2"],
            "data_requirements": ["some_req"],
            "missing_data_requirements": ["some_missing_req"]
        }

        # classify_intent_node directly handles "SYSTEM_CLEAR_HISTORY"
        # so no LLM call for intent classification is expected for this specific input.
        # If it were a natural language "clear data", then LLM would be called.

        final_state = self.workflow.invoke(initial_state_for_clear, config=config)

        self.assertEqual(final_state["input"], user_input) # Input preserved
        self.assertEqual(final_state["intent_list"], ["clear_data_intent"]) # Set by classify_intent_node
        self.assertEqual(final_state["latest_df"], collections.OrderedDict())
        self.assertEqual(final_state["df_history"], [])
        self.assertIsNone(final_state["SQL"])
        self.assertEqual(final_state["interpretation"], DATA_CLEARED_MESSAGE)
        self.assertIsNone(final_state["chart_result"])
        self.assertEqual(final_state["metadata_answer"], "Old metadata to be preserved") # Preserved
        self.assertIsNone(final_state["error"])
        self.assertEqual(final_state["query_history"], []) # Cleared by clear_data_node
        self.assertEqual(final_state["condition"], "データクリア完了")

        # data_requirements and missing_data_requirements are not explicitly managed by clear_data_node
        # Their state would depend on whether they were part of the input to clear_data_node
        # and if clear_data_node explicitly clears them or if they are cleared by virtue of
        # not being passed forward from the preserved fields.
        # Based on clear_data_node, they are not explicitly preserved or cleared, so they will be absent
        # if not part of the minimal set of keys returned by clear_data_node.
        self.assertNotIn("data_requirements", final_state) # Or assert to specific cleared state if intended
        self.assertNotIn("missing_data_requirements", final_state)


    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.extract_sql', side_effect=lambda x: x.replace("```sql", "").replace("```", "").strip()) # Simple mock for extract_sql
    @patch('files.backend_codes.uuid.uuid4')
    def test_workflow_sql_error_recovery_success(
        self, mock_uuid, mock_extract_sql, mock_try_sql, mock_vs_queries, mock_vs_tables, mock_llm
    ):
        user_input = "Show New Product Sales"
        config = {"configurable": {"thread_id": "test_sql_recovery_success"}}
        mock_uuid.return_value = MagicMock(hex="rec_ok_".ljust(8, '0'))

        # LLM responses
        llm_responses = [
            MagicMock(content="データ取得,データ解釈"),                 # 1. classify_intent
            MagicMock(content="New Product Sales"),               # 2. extract_data_requirements
            MagicMock(content="```sql\nSELECT * FRMO new_product_sales;\n```"), # 3. sql_node (initial faulty SQL)
            MagicMock(content="```sql\nSELECT * FROM new_product_sales;\n```"), # 4. fix_sql_with_llm (corrected SQL)
            MagicMock(content="Interpretation of new product sales.") # 5. interpret_node
        ]
        mock_llm.invoke.side_effect = llm_responses

        # DB execution results
        db_results = [
            (None, "syntax error near FRMO"), # 1. For faulty SQL
            (pd.DataFrame([{"product": "SuperWidget", "sales": 150}]), None) # 2. For corrected SQL
        ]
        mock_try_sql.side_effect = db_results

        # RAG (assume no relevant history for "New Product Sales")
        mock_vs_tables.return_value = [MagicMock(page_content="Table: new_product_sales (product, sales)")]
        mock_vs_queries.return_value = []


        initial_workflow_state = {"input": user_input, "df_history": []}
        final_state = self.workflow.invoke(initial_workflow_state, config=config)

        self.assertEqual(final_state["condition"], "解釈完了")
        self.assertIn("New Product Sales", final_state["latest_df"])
        self.assertEqual(final_state["latest_df"]["New Product Sales"], [{"product": "SuperWidget", "sales": 150}])
        self.assertEqual(final_state["interpretation"], "Interpretation of new product sales.")
        self.assertIsNone(final_state["error"])
        self.assertEqual(len(final_state["df_history"]), 1)
        self.assertEqual(final_state["df_history"][0]["query"], "New Product Sales")
        self.assertEqual(final_state["df_history"][0]["SQL"], "SELECT * FROM new_product_sales;")

        self.assertEqual(mock_llm.invoke.call_count, 5)
        self.assertEqual(mock_try_sql.call_count, 2)
        # extract_sql is called by sql_node for initial gen and by fix_sql_with_llm
        # sql_node calls it once after the initial SQL generation, and fix_sql_with_llm calls it once.
        self.assertEqual(mock_extract_sql.call_count, 2)


    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.extract_sql', side_effect=lambda x: x.replace("```sql", "").replace("```", "").strip())
    # transform_sql_error is NOT mocked to test its actual behavior.
    def test_workflow_sql_error_recovery_failure(
        self, mock_extract_sql, mock_try_sql, mock_vs_queries, mock_vs_tables, mock_llm
    ):
        user_input = "Show Problematic Sales Data"
        config = {"configurable": {"thread_id": "test_sql_recovery_fail"}}

        llm_responses = [
            MagicMock(content="データ取得"),       # 1. classify_intent
            MagicMock(content="Problematic Sales"), # 2. extract_data_requirements
            MagicMock(content="```sql\nSELECT * FRMO problematic_sales;\n```"), # 3. sql_node (initial faulty SQL)
            MagicMock(content="```sql\nSELECT * FRRMO problematic_sales;\n```") # 4. fix_sql_with_llm (still faulty)
        ]
        mock_llm.invoke.side_effect = llm_responses

        db_results = [
            (None, "syntax error near FRMO"),    # 1. For initial faulty SQL
            (None, "syntax error near FRRMO")    # 2. For "fixed" faulty SQL
        ]
        mock_try_sql.side_effect = db_results

        mock_vs_tables.return_value = [MagicMock(page_content="Table: problematic_sales (id, value)")]
        mock_vs_queries.return_value = []

        initial_workflow_state = {"input": user_input, "df_history": []}
        final_state = self.workflow.invoke(initial_workflow_state, config=config)

        self.assertEqual(final_state["condition"], "SQL部分的失敗") # Or SQL実行失敗 if it's the only req
        self.assertIsNotNone(final_state["error"])
        # Check that the error message is the user-friendly one from transform_sql_error
        expected_user_friendly_error = "SQL構文にエラーがあります。クエリを確認してください。(詳細: syntax error near FRRMO)"
        self.assertIn(expected_user_friendly_error, final_state["error"])

        self.assertNotIn("Problematic Sales", final_state.get("latest_df", {}))
        self.assertEqual(final_state["SQL"], "SELECT * FRRMO problematic_sales;") # Last attempted SQL

        self.assertEqual(mock_llm.invoke.call_count, 4)
        self.assertEqual(mock_try_sql.call_count, 2)
        self.assertEqual(mock_extract_sql.call_count, 2)


class TestClassifyIntentNode(unittest.TestCase):

    @patch('files.backend_codes.llm')
    def test_classify_intent_metadata_table_question(self, mock_llm):
        user_input = "sales_dataテーブルにはどんなカラムがありますか？"
        expected_intent_str = "メタデータ検索"
        mock_llm.invoke.return_value = MagicMock(content=expected_intent_str)

        initial_state = MyState(input=user_input, query_history=[])
        result_state = classify_intent_node(initial_state)

        mock_llm.invoke.assert_called_once()
        # Verify prompt structure if necessary, though it's standard for this node
        prompt_arg = mock_llm.invoke.call_args[0][0]
        self.assertIn(user_input, prompt_arg)
        self.assertIn("ユーザーの質問の意図を判定してください。", prompt_arg)

        self.assertEqual(result_state["intent_list"], [expected_intent_str])
        self.assertEqual(result_state["condition"], "分類完了")
        self.assertEqual(result_state["query_history"], [user_input])

    @patch('files.backend_codes.llm')
    def test_classify_intent_metadata_column_question(self, mock_llm):
        user_input = "categoryカラムの情報を教えてください"
        expected_intent_str = "メタデータ検索"
        mock_llm.invoke.return_value = MagicMock(content=expected_intent_str)

        initial_state = MyState(input=user_input, query_history=["previous query"])
        result_state = classify_intent_node(initial_state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["intent_list"], [expected_intent_str])
        self.assertEqual(result_state["condition"], "分類完了")
        self.assertEqual(result_state["query_history"], ["previous query", user_input])

    @patch('files.backend_codes.llm')
    def test_classify_intent_regular_data_query(self, mock_llm):
        user_input = "カテゴリごとの売上を知りたい"
        expected_intent_str = "データ取得"
        mock_llm.invoke.return_value = MagicMock(content=expected_intent_str)

        initial_state = MyState(input=user_input) # query_history will be initialized
        result_state = classify_intent_node(initial_state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["intent_list"], [expected_intent_str])
        self.assertEqual(result_state["condition"], "分類完了")
        self.assertEqual(result_state["query_history"], [user_input])

    @patch('files.backend_codes.llm')
    def test_classify_intent_greeting(self, mock_llm):
        user_input = "こんにちは"
        # Assuming LLM might return empty or a non-intent for a greeting
        expected_intent_str = ""
        mock_llm.invoke.return_value = MagicMock(content=expected_intent_str)

        initial_state = MyState(input=user_input, query_history=[])
        result_state = classify_intent_node(initial_state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["intent_list"], []) # Empty string splits to empty list effectively
        self.assertEqual(result_state["condition"], "分類完了")
        self.assertEqual(result_state["query_history"], [user_input])

    @patch('files.backend_codes.llm')
    def test_classify_intent_multiple_intents(self, mock_llm):
        user_input = "A商品の売上を取得してグラフ化し、その結果を解釈して"
        expected_intent_str = "データ取得,グラフ作成,データ解釈"
        mock_llm.invoke.return_value = MagicMock(content=expected_intent_str)

        initial_state = MyState(input=user_input, query_history=[])
        result_state = classify_intent_node(initial_state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["intent_list"], ["データ取得", "グラフ作成", "データ解釈"])
        self.assertEqual(result_state["condition"], "分類完了")
        self.assertEqual(result_state["query_history"], [user_input])

    def test_classify_intent_system_clear_history(self):
        # This case does not call the LLM.
        user_input = "SYSTEM_CLEAR_HISTORY"
        initial_state = MyState(input=user_input, query_history=["previous query"])
        result_state = classify_intent_node(initial_state)

        self.assertEqual(result_state["intent_list"], ["clear_data_intent"])
        self.assertEqual(result_state["condition"], "分類完了")
        # query_history should be preserved up to this point by classify_intent_node
        # before clear_data_node actually clears it.
        self.assertEqual(result_state["query_history"], ["previous query"])


# In TestMetadataRetrievalNode, add tests based on test_metadata_feature.py
@patch('files.backend_codes.vectorstore_tables.similarity_search')
@patch('files.backend_codes.llm')
def add_new_metadata_tests(self, mock_llm, mock_vs_tables_search): # Placeholder to inject tests

    # Test Case: Metadata Retrieval for a general table question (e.g., 'users' table)
    user_input_users = "usersテーブルについて教えて"
    mock_rag_docs_users = [
        MagicMock(page_content="Table: users Columns: id (INTEGER), name (TEXT), email (TEXT), created_at (TEXT)"),
        MagicMock(page_content="Table: users Description: Stores user information.")
    ]
    mock_vs_tables_search.return_value = mock_rag_docs_users
    expected_llm_answer_users = "usersテーブルはユーザー情報を格納し、id, name, email, created_atカラムがあります。"
    mock_llm.invoke.return_value = MagicMock(content=expected_llm_answer_users)

    state_users = MyState(input=user_input_users)
    result_state_users = metadata_retrieval_node(state_users)

    mock_vs_tables_search.assert_called_with(user_input_users, k=3)
    llm_prompt_users = str(mock_llm.invoke.call_args[0][0])
    self.assertIn(user_input_users, llm_prompt_users)
    self.assertIn(mock_rag_docs_users[0].page_content, llm_prompt_users)
    self.assertIn(mock_rag_docs_users[1].page_content, llm_prompt_users)
    self.assertEqual(result_state_users["metadata_answer"], expected_llm_answer_users)
    self.assertEqual(result_state_users["condition"], "メタデータ検索完了")
    mock_llm.reset_mock() # Reset for next sub-test
    mock_vs_tables_search.reset_mock()

    # Test Case: Metadata Retrieval for a specific column question (e.g., 'products' table's 'price' column)
    user_input_price = "productsテーブルのpriceカラムは何ですか"
    mock_rag_docs_price = [
        MagicMock(page_content="Table: products Columns: id (INTEGER), name (TEXT), price (REAL)"),
        MagicMock(page_content="Column: price (REAL) in products table stores the product price as a numeric value.")
    ]
    mock_vs_tables_search.return_value = mock_rag_docs_price
    expected_llm_answer_price = "productsテーブルのpriceカラムは商品の価格を数値で格納します。"
    mock_llm.invoke.return_value = MagicMock(content=expected_llm_answer_price)

    state_price = MyState(input=user_input_price)
    result_state_price = metadata_retrieval_node(state_price)

    mock_vs_tables_search.assert_called_with(user_input_price, k=3)
    llm_prompt_price = str(mock_llm.invoke.call_args[0][0])
    self.assertIn(user_input_price, llm_prompt_price)
    self.assertIn(mock_rag_docs_price[0].page_content, llm_prompt_price)
    self.assertIn(mock_rag_docs_price[1].page_content, llm_prompt_price)
    self.assertEqual(result_state_price["metadata_answer"], expected_llm_answer_price)
    self.assertEqual(result_state_price["condition"], "メタデータ検索完了")
    mock_llm.reset_mock()
    mock_vs_tables_search.reset_mock()

    # Test Case: Metadata Retrieval for a vague question
    user_input_vague = "顧客データについて"
    mock_rag_docs_vague = [
        MagicMock(page_content="Table: customers Columns: id, name, address, loyalty_points"),
        MagicMock(page_content="Table: users Columns: user_id, username, email (related to customers)")
    ]
    mock_vs_tables_search.return_value = mock_rag_docs_vague
    expected_llm_answer_vague = "顧客データとしてはcustomersテーブルやusersテーブルがあり、顧客の基本情報やロイヤルティポイント、ユーザーアカウント情報などが含まれます。"
    mock_llm.invoke.return_value = MagicMock(content=expected_llm_answer_vague)

    state_vague = MyState(input=user_input_vague)
    result_state_vague = metadata_retrieval_node(state_vague)

    mock_vs_tables_search.assert_called_with(user_input_vague, k=3)
    llm_prompt_vague = str(mock_llm.invoke.call_args[0][0])
    self.assertIn(user_input_vague, llm_prompt_vague)
    self.assertIn(mock_rag_docs_vague[0].page_content, llm_prompt_vague)
    self.assertIn(mock_rag_docs_vague[1].page_content, llm_prompt_vague)
    self.assertEqual(result_state_vague["metadata_answer"], expected_llm_answer_vague)
    self.assertEqual(result_state_vague["condition"], "メタデータ検索完了")

# Dynamically add the new tests to TestMetadataRetrievalNode
TestMetadataRetrievalNode.test_metadata_retrieval_general_table_question = add_new_metadata_tests
TestMetadataRetrievalNode.test_metadata_retrieval_specific_column_question = add_new_metadata_tests
TestMetadataRetrievalNode.test_metadata_retrieval_vague_question = add_new_metadata_tests
# Note: The "non-existent table" case is already well-covered by test_metadata_retrieval_no_rag_docs_found.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
