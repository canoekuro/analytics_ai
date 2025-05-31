import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import collections
import uuid
from datetime import datetime

# Assuming backend_codes.py is in the same directory or accessible via PYTHONPATH
from files.backend_codes import (
    MyState,
    extract_data_requirements_node,
    find_similar_query_node,
    sql_node,
    build_workflow,
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

    def test_some_requirements_missing(self):
        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の売上集計", "timestamp": "ts1", # Matches "A商品の売上" well
            "dataframe_dict": [{"product": "A", "sales": 100}], "SQL": "SQL1"
        }
        # This setup assumes SIMILARITY_THRESHOLD in backend_codes.py is around 0.6-0.8 for this to work.
        # "A商品の売上" vs "A商品の売上集計" should be high.
        # "B商品の在庫" vs "A商品の売上集計" should be low.
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

    def test_history_entry_used_once(self):
        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の詳細データ", "timestamp": "ts1",
            "dataframe_dict": [{"product": "A", "detail": "very detailed"}], "SQL": "SQL_A_detail"
        }
        state = MyState(
            input="A商品の詳細とA商品の情報", # Two requirements
            data_requirements=["A商品の詳細", "A商品の情報"],
            df_history=[sample_history_entry_1], # Only one relevant history entry
            intent_list=["データ取得"]
        )
        # backend_codes.SIMILARITY_THRESHOLD will determine if "A商品の情報" matches "A商品の詳細データ"
        # Assuming "A商品の詳細" is a better or earlier match.

        result_state = find_similar_query_node(state)

        self.assertEqual(result_state["condition"], "missing_data")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 1) # Only one should be found

        # Verify which one was found and which is missing
        if "A商品の詳細" in result_state["latest_df"]:
            self.assertEqual(result_state["latest_df"]["A商品の詳細"], [{"product": "A", "detail": "very detailed"}])
            self.assertIn("A商品の情報", result_state["missing_data_requirements"])
        elif "A商品の情報" in result_state["latest_df"]: # This case depends on similarity scoring
            self.assertEqual(result_state["latest_df"]["A商品の情報"], [{"product": "A", "detail": "very detailed"}])
            self.assertIn("A商品の詳細", result_state["missing_data_requirements"])
        else:
            self.fail("The found data in latest_df does not match any of the expected requirements.")


class TestSqlNode(unittest.TestCase):

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4') # To control history IDs
    def test_sql_node_with_missing_requirements(
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
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


# Import clear_data_node for testing
from files.backend_codes import clear_data_node, DATA_CLEARED_MESSAGE

class TestClearDataNode(unittest.TestCase):
    def test_clear_data_node_resets_state_correctly(self):
        initial_state = MyState(
            input="リセットして",
            intent_list=["clear_data_intent"],
            latest_df=collections.OrderedDict({"some_data": [{"col": "val"}]}),
            df_history=[{"id": "old_id", "query": "old_query", "dataframe_dict": [], "SQL": "OLD SQL"}],
            SQL="SELECT * FROM old_table",
            interpretation="Old interpretation",
            chart_result="old_chart_base64",
            metadata_answer="Old metadata", # This should be preserved as per current clear_data_node
            condition="some_condition",
            error="some_error",
            query_history=["q1", "q2"]
        )

        cleared_state = clear_data_node(initial_state)

        self.assertEqual(cleared_state["input"], "リセットして") # Preserved
        self.assertEqual(cleared_state["intent_list"], ["clear_data_intent"]) # Preserved
        self.assertEqual(cleared_state["latest_df"], collections.OrderedDict()) # Reset to empty OrderedDict
        self.assertEqual(cleared_state["df_history"], []) # Cleared
        self.assertIsNone(cleared_state["SQL"]) # Cleared
        self.assertEqual(cleared_state["interpretation"], DATA_CLEARED_MESSAGE) # Set to cleared message
        self.assertIsNone(cleared_state["chart_result"]) # Cleared
        self.assertEqual(cleared_state["metadata_answer"], "Old metadata") # Preserved
        self.assertEqual(cleared_state["condition"], "データクリア完了")
        self.assertIsNone(cleared_state["error"]) # Cleared
        self.assertEqual(cleared_state["query_history"], []) # Cleared


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

        # Ensure mocks for nodes not reached (sql_node, chart_node) were not called
        mock_vs_tables.assert_not_called()
        mock_vs_queries.assert_not_called()
        mock_try_sql.assert_not_called()
        mock_repl_tool.assert_not_called()

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.try_sql_execute')
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
        # 4. chart_node: Python code for charting (not checking exact code, just that it's called)
        mock_llm.invoke.side_effect = [
            MagicMock(content="データ取得,グラフ作成"),
            MagicMock(content="A商品の売上,B商品の在庫"),
            MagicMock(content="SELECT * FROM B_inventory;"), # SQL for B商品の在庫 in sql_node
            MagicMock(content="print('Chart generated')") # Python code from chart agent
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
        mock_repl_tool_instance.invoke.return_value = "Chart tool executed" # Simulate tool output
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
            final_state = self.workflow.invoke(
                {"input": user_input, "df_history": [history_entry_A]}, # df_history for the start
                config=config
            )

        # --- Assertions ---
        # Path: classify -> extract_data -> find_similar (missing) -> sql -> chart -> END
        self.assertEqual(final_state["condition"], "グラフ化完了")
        self.assertEqual(final_state["chart_result"], "encoded_chart_data")

        # Check latest_df has both A (from history via find_similar) and B (from sql_node)
        self.assertIn("A商品の売上", final_state["latest_df"])
        self.assertEqual(final_state["latest_df"]["A商品の売上"], data_A_sales)
        self.assertIn("B商品の在庫", final_state["latest_df"])
        self.assertEqual(final_state["latest_df"]["B商品の在庫"], data_B_inventory_df.to_dict(orient="records"))

        # Check history updates from sql_node
        self.assertTrue(any(h["query"] == "B商品の在庫" for h in final_state["df_history"]))

        # LLM Calls: classify, extract_req, sql_gen_B, chart_agent_prompt
        self.assertEqual(mock_llm.invoke.call_count, 4)

        # RAG calls for B商品の在庫 in sql_node
        mock_vs_tables.assert_called_once_with("B商品の在庫", k=3)
        mock_vs_queries.assert_called_once_with("B商品の在庫", k=3)

        # DB call for B商品の在庫 in sql_node
        mock_try_sql.assert_called_once_with("SELECT * FROM B_inventory;")

        # Chart tool call
        mock_repl_tool_instance.invoke.assert_called_once()
        mock_path_exists.assert_called_once_with("output.png")
        mock_b64encode.assert_called_once_with(b"chart_bytes")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
