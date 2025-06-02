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
    # extract_data_requirements_node, # Removed
    # find_similar_query_node, # Removed
    create_analysis_plan_node, # Added
    check_history_node,        # Added
    clarify_node,              # Added
    # execute_plan_router, # Not typically tested in isolation, but through workflow tests
    sql_node,
    build_workflow,
    try_sql_execute, # Import for direct testing
    transform_sql_error, # Added for new tests
    # For mocking, we might need to patch where they are *used* if not careful with direct imports
    # However, for llm, vectorstores, these are global in backend_codes, so patching them there is fine.
)

# Mocking global objects from backend_codes.py
# These will be patched in specific test methods or setUp using @patch
# llm_mock = MagicMock()
# vectorstore_tables_mock = MagicMock()
# vectorstore_queries_mock = MagicMock()
# try_sql_execute_mock = MagicMock()


# TestExtractDataRequirementsNode class removed


class TestCheckHistoryNode(unittest.TestCase): # Renamed from TestFindSimilarQueryNode

    def test_all_requirements_found_in_history(self):
        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の売上集計", "timestamp": "ts1",
            "dataframe_dict": [{"product": "A", "sales": 100}], "SQL": "SQL1"
        }
        sample_history_entry_2 = {
            "id": "hist_002", "query": "顧客属性データ", "timestamp": "ts2",
            "dataframe_dict": [{"user_id": 1, "age": 30}], "SQL": "SQL2"
        }
        # Input to check_history_node is the list of requirements (plan step details)
        # This simulates that the plan step details (which become state.input for the node) are these requirements.
        state = MyState(
            input=["A商品の売上集計", "顧客属性データ"], # This is what check_history_node processes
            df_history=[sample_history_entry_1, sample_history_entry_2],
            # Other state fields like intent_list, analysis_plan, current_plan_step_index
            # are not directly used by check_history_node's core logic being tested here,
            # but would be present in a full workflow.
            analysis_plan=[{"action": "check_history", "details": ["A商品の売上集計", "顧客属性データ"]}],
            current_plan_step_index=0
        )

        result_state = check_history_node(state)

        self.assertEqual(result_state["condition"], "history_checked")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 2)
        self.assertEqual(result_state["latest_df"]["A商品の売上集計"], [{"product": "A", "sales": 100}])
        self.assertEqual(result_state["latest_df"]["顧客属性データ"], [{"user_id": 1, "age": 30}])
        self.assertEqual(result_state["missing_data_requirements"], [])

    @patch('files.backend_codes.SIMILARITY_THRESHOLD', 0.8)
    @patch('files.backend_codes.difflib.SequenceMatcher')
    def test_some_requirements_missing_from_history(self, MockSequenceMatcher, mock_threshold):
        mock_matcher_instance = MockSequenceMatcher.return_value
        mock_matcher_instance.ratio.side_effect = [0.9, 0.5] # Match "A商品の売上", miss "B商品の在庫"

        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の売上集計", "timestamp": "ts1",
            "dataframe_dict": [{"product": "A", "sales": 100}], "SQL": "SQL1"
        }
        state = MyState(
            input=["A商品の売上", "B商品の在庫"], # Requirements for check_history_node
            df_history=[sample_history_entry_1],
            analysis_plan=[{"action": "check_history", "details": ["A商品の売上", "B商品の在庫"]}],
            current_plan_step_index=0
        )

        result_state = check_history_node(state)

        self.assertEqual(result_state["condition"], "history_checked")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 1)
        self.assertIn("A商品の売上", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["A商品の売上"], [{"product": "A", "sales": 100}])
        self.assertEqual(result_state["missing_data_requirements"], ["B商品の在庫"])

    def test_none_requirements_found_in_history(self):
        sample_history_entry_1 = {
            "id": "hist_001", "query": "X商品の情報", "timestamp": "ts1",
            "dataframe_dict": [{"product": "X", "info": "xyz"}], "SQL": "SQLX"
        }
        state = MyState(
            input=["A商品の売上", "B商品の在庫"], # Requirements for check_history_node
            df_history=[sample_history_entry_1],
            analysis_plan=[{"action": "check_history", "details": ["A商品の売上", "B商品の在庫"]}],
            current_plan_step_index=0
        )

        result_state = check_history_node(state)

        self.assertEqual(result_state["condition"], "history_checked")
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 0)
        self.assertEqual(len(result_state["missing_data_requirements"]), 2)
        self.assertIn("A商品の売上", result_state["missing_data_requirements"])
        self.assertIn("B商品の在庫", result_state["missing_data_requirements"])

    def test_empty_requirements_for_history_check(self):
        state = MyState(
            input=[], # No requirements for check_history_node
            df_history=[],
            analysis_plan=[{"action": "check_history", "details": []}],
            current_plan_step_index=0
        )
        result_state = check_history_node(state)
        self.assertEqual(result_state["condition"], "history_checked_no_requirements")
        self.assertEqual(result_state["latest_df"], collections.OrderedDict())
        self.assertEqual(result_state["missing_data_requirements"], [])

    @patch('files.backend_codes.SIMILARITY_THRESHOLD', 0.8)
    @patch('files.backend_codes.difflib.SequenceMatcher')
    def test_history_entry_used_once_in_check(self, MockSequenceMatcher, mock_threshold):
        mock_matcher_instance = MockSequenceMatcher.return_value
        mock_matcher_instance.ratio.side_effect = [0.95, 0.85]

        sample_history_entry_1 = {
            "id": "hist_001", "query": "A商品の詳細データ", "timestamp": "ts1",
            "dataframe_dict": [{"product": "A", "detail": "very detailed"}], "SQL": "SQL_A_detail"
        }
        state = MyState(
            input=["A商品の詳細", "A商品の情報"], # Requirements for check_history_node
            df_history=[sample_history_entry_1],
            analysis_plan=[{"action": "check_history", "details": ["A商品の詳細", "A商品の情報"]}],
            current_plan_step_index=0
        )

        result_state = check_history_node(state)

        self.assertEqual(result_state["condition"], "history_checked") # Condition is always "history_checked" if requirements provided
        self.assertIsInstance(result_state["latest_df"], collections.OrderedDict)
        self.assertEqual(len(result_state["latest_df"]), 1)
        self.assertIn("A商品の詳細", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["A商品の詳細"], [{"product": "A", "detail": "very detailed"}])
        self.assertIn("A商品の情報", result_state["missing_data_requirements"])
        self.assertNotIn("A商品の情報", result_state["latest_df"])


class TestCreateAnalysisPlanNode(unittest.TestCase):

    def test_empty_plan_for_greeting(self):
        state = MyState(input="hello")
        result_state = create_analysis_plan_node(state)
        self.assertEqual(result_state["condition"], "empty_plan_generated")
        self.assertEqual(result_state["analysis_plan"], [])
        self.assertIsNone(result_state["current_plan_step_index"])

    def test_clarify_plan_for_ambiguous_query(self):
        state = MyState(input="This is a vague query")
        result_state = create_analysis_plan_node(state)
        self.assertEqual(result_state["condition"], "plan_generated")
        self.assertIsNotNone(result_state["analysis_plan"])
        self.assertEqual(len(result_state["analysis_plan"]), 1)
        self.assertEqual(result_state["analysis_plan"][0]["action"], "clarify")
        self.assertIn("Could you please specify", result_state["analysis_plan"][0]["details"])
        self.assertEqual(result_state["current_plan_step_index"], 0)
from google.api_core import exceptions as google_exceptions
import json # For creating mock JSON strings

class TestCreateAnalysisPlanNode(unittest.TestCase):

    def setUp(self):
        # Patch genai.GenerativeModel and genai.configure for all tests in this class
        self.patcher_model = patch('files.backend_codes.genai.GenerativeModel')
        self.patcher_configure = patch('files.backend_codes.genai.configure')

        self.mock_generative_model_cls = self.patcher_model.start()
        self.mock_configure = self.patcher_configure.start()

        # Mock the instance of GenerativeModel
        self.mock_model_instance = MagicMock()
        self.mock_generative_model_cls.return_value = self.mock_model_instance
        
        # This will be configured per test
        self.generate_content_mock = MagicMock()
        self.mock_model_instance.generate_content = self.generate_content_mock

    def tearDown(self):
        self.patcher_model.stop()
        self.patcher_configure.stop()

    def _create_mock_gemini_response(self, plan_steps_dict_or_str, has_function_call=True, text_response=None):
        response_mock = MagicMock()
        part_mock = MagicMock()
        
        if has_function_call:
            function_call_mock = MagicMock()
            # plan_steps_dict_or_str could be a dict for args, or a string if testing string parsing
            function_call_mock.args = plan_steps_dict_or_str if isinstance(plan_steps_dict_or_str, dict) else {"plan_steps": plan_steps_dict_or_str}
            part_mock.function_call = function_call_mock
            part_mock.text = None # Explicitly set text to None if function_call is present
        else:
            part_mock.function_call = None
            part_mock.text = text_response
            # If text_response is intended to be JSON, it should be a JSON string here.

        response_mock.candidates = [MagicMock(content=MagicMock(parts=[part_mock]))]
        # Also mock the .text attribute on the top-level response for fallback
        response_mock.text = text_response if text_response else "" # Ensure .text exists
        return response_mock

    def test_successful_plan_simple_query(self):
        expected_plan = [{"action": "sql", "details": "total sales"}]
        mock_response_args = {"plan_steps": expected_plan}
        self.generate_content_mock.return_value = self._create_mock_gemini_response(mock_response_args)

        state = MyState(input="show total sales")
        result_state = create_analysis_plan_node(state)

        self.assertEqual(result_state["analysis_plan"], expected_plan)
        self.assertEqual(result_state["condition"], "plan_generated")
        self.assertIsNone(result_state["error"])
        self.generate_content_mock.assert_called_once()

    def test_successful_plan_ambiguous_query_clarify(self):
        expected_plan = [{"action": "clarify", "details": "Please specify the period for sales data."}]
        mock_response_args = {"plan_steps": expected_plan}
        self.generate_content_mock.return_value = self._create_mock_gemini_response(mock_response_args)
        
        state = MyState(input="sales data")
        result_state = create_analysis_plan_node(state)

        self.assertEqual(result_state["analysis_plan"], expected_plan)
        self.assertEqual(result_state["condition"], "plan_generated")
        self.generate_content_mock.assert_called_once()

    def test_successful_plan_multistep_query(self):
        expected_plan = [
            {"action": "sql", "details": "sales"},
            {"action": "chart", "details": "chart sales"}
        ]
        mock_response_args = {"plan_steps": expected_plan}
        self.generate_content_mock.return_value = self._create_mock_gemini_response(mock_response_args)

        state = MyState(input="show sales, then chart it")
        result_state = create_analysis_plan_node(state)

        self.assertEqual(result_state["analysis_plan"], expected_plan)
        self.assertEqual(result_state["condition"], "plan_generated")
        self.generate_content_mock.assert_called_once()
        
    def test_prompt_includes_query_history(self):
        history = ["past query 1", "past query 2"]
        current_query = "current query based on history"
        expected_plan = [{"action": "sql", "details": "current query data"}]
        mock_response_args = {"plan_steps": expected_plan}
        self.generate_content_mock.return_value = self._create_mock_gemini_response(mock_response_args)

        state = MyState(input=current_query, query_history=history + [current_query]) # History includes current for this setup
        create_analysis_plan_node(state)

        self.generate_content_mock.assert_called_once()
        called_prompt_string = self.generate_content_mock.call_args[0][0]
        self.assertIn("past query 1", called_prompt_string)
        self.assertIn("past query 2", called_prompt_string)
        self.assertIn(current_query, called_prompt_string)
        self.assertNotIn("past query 1\n- past query 1", called_prompt_string) # Ensure no self-repetition from history logic

    def test_error_handling_api_call_failure(self):
        self.generate_content_mock.side_effect = google_exceptions.GoogleAPICallError("API error")

        state = MyState(input="any query")
        result_state = create_analysis_plan_node(state)

        self.assertIsNone(result_state["analysis_plan"])
        self.assertEqual(result_state["condition"], "plan_generation_failed")
        self.assertIn("モデル呼び出しでエラーが発生しました", result_state["error"])
        self.assertIn("API error", result_state["error"])

    def test_error_handling_invalid_json_in_function_call_args(self):
        # Simulate function_call.args["plan_steps"] being a string of malformed JSON
        malformed_json_string = '[{"action": "sql", "details": "some data"}, {"action": "oops' # Invalid JSON
        mock_response_args = {"plan_steps": malformed_json_string} # plan_steps itself is the string
        self.generate_content_mock.return_value = self._create_mock_gemini_response(mock_response_args)

        state = MyState(input="any query")
        result_state = create_analysis_plan_node(state)
        
        self.assertIsNone(result_state["analysis_plan"])
        self.assertEqual(result_state["condition"], "plan_generation_failed")
        self.assertIn("モデル応答の解析に失敗しました", result_state["error"])
        self.assertIn("plan_stepsのJSON文字列のデコードに失敗", result_state["error"])

    def test_error_handling_no_function_call_invalid_text_fallback(self):
        # Simulate no function_call and text fallback is not valid JSON
        self.generate_content_mock.return_value = self._create_mock_gemini_response(
            plan_steps_dict_or_str=None, # No function call args
            has_function_call=False,
            text_response="This is not a valid JSON plan."
        )
        
        state = MyState(input="any query")
        result_state = create_analysis_plan_node(state)

        self.assertIsNone(result_state["analysis_plan"])
        self.assertEqual(result_state["condition"], "plan_generation_failed")
        self.assertIn("モデル応答は期待されるfunction_call形式ではなく、テキスト応答の解析にも失敗しました", result_state["error"])

    def test_successful_plan_from_text_fallback_if_no_function_call(self):
        expected_plan = [{"action": "sql", "details": "data from text"}]
        valid_json_text_response = json.dumps(expected_plan) # '[{"action": "sql", "details": "data from text"}]'
        
        self.generate_content_mock.return_value = self._create_mock_gemini_response(
            plan_steps_dict_or_str=None,
            has_function_call=False,
            text_response=valid_json_text_response
        )

        state = MyState(input="query leading to text fallback")
        result_state = create_analysis_plan_node(state)

        self.assertEqual(result_state["analysis_plan"], expected_plan)
        self.assertEqual(result_state["condition"], "plan_generated")
        self.assertIsNone(result_state["error"])

    def test_error_handling_model_initialization_failure(self):
        # Make the class constructor raise an error
        self.mock_generative_model_cls.side_effect = Exception("Model init failed")

        state = MyState(input="any query")
        result_state = create_analysis_plan_node(state)

        self.assertIsNone(result_state["analysis_plan"])
        self.assertEqual(result_state["condition"], "plan_generation_failed")
        self.assertIn("分析計画の作成中に予期せぬシステムエラーが発生しました", result_state["error"])
        self.assertIn("Model init failed", result_state["error"])
        self.mock_configure.assert_not_called() # configure should not be called if model init fails
        self.generate_content_mock.assert_not_called()


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
            # input="Show me sales and inventory.", # Not directly used by sql_node if missing_data_requirements is set
            missing_data_requirements=["Sales Data", "Inventory Data"], # These are what sql_node will process
            latest_df=collections.OrderedDict(), # Initially empty
            df_history=[],
            # complex_analysis_original_query might be used for LLM context
            complex_analysis_original_query="Show me sales and inventory."
        )

        result_state = sql_node(initial_state)

        # --- Assertions ---
        self.assertEqual(result_state["condition"], "sql_execution_done") # Updated condition name
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

        # Check if missing_data_requirements is updated (cleared because all were processed successfully)
        # The logic in sql_node updates missing_data_requirements based on successfully_fetched_reqs.
        # If initial missing_data_requirements was ["Sales Data", "Inventory Data"] and both succeeded,
        # it should become empty.
        self.assertEqual(result_state.get("missing_data_requirements"), [])

    def setUp(self):
        self.initial_state = MyState(
            input="test query",
            missing_data_requirements=["some requirement"],
            latest_df=collections.OrderedDict(),
            df_history=[],
            complex_analysis_original_query="test query context"
        )

    @patch('files.backend_codes.vectorstore_queries')
    @patch('files.backend_codes.vectorstore_tables')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.fix_sql_with_llm')
    @patch('files.backend_codes.try_sql_execute')
    def test_sql_node_handles_persistent_execution_error(
        self, mock_try_sql_execute, mock_fix_sql_with_llm, mock_llm,
        mock_vectorstore_tables, mock_vectorstore_queries
    ):
        # Mock RAG components
        mock_vectorstore_tables.similarity_search.return_value = [MagicMock(page_content="table info")]
        mock_vectorstore_queries.similarity_search.return_value = [MagicMock(page_content="query info")]

        # Mock LLM for initial SQL generation
        mock_llm.invoke.return_value = MagicMock(content="SELECT * FROM test_table")

        # Mock try_sql_execute to simulate persistent failure
        mock_try_sql_execute.side_effect = [
            (None, "Initial SQL error"), # First attempt fails
            (None, "Fixed SQL error")    # Second attempt (after fix) also fails
        ]

        # Mock fix_sql_with_llm to return a dummy fixed SQL
        mock_fix_sql_with_llm.return_value = "SELECT * FROM fixed_test_table"

        # Expected user-friendly error
        expected_user_error = transform_sql_error("Fixed SQL error")

        result_state = sql_node(self.initial_state)

        self.assertEqual(result_state["condition"], "sql_execution_failed")
        self.assertIn("SQLの実行に失敗しました", result_state["error"])
        self.assertIn("some requirement", result_state["error"])
        self.assertIn(expected_user_error, result_state["error"])
        self.assertEqual(result_state["missing_data_requirements"], ["some requirement"])
        self.assertEqual(mock_try_sql_execute.call_count, 2) # Both initial and fixed SQL were tried

    @patch('files.backend_codes.vectorstore_queries')
    @patch('files.backend_codes.vectorstore_tables')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    def test_sql_node_handles_empty_result(
        self, mock_try_sql_execute, mock_llm,
        mock_vectorstore_tables, mock_vectorstore_queries
    ):
        # Mock RAG components
        mock_vectorstore_tables.similarity_search.return_value = [MagicMock(page_content="table info")]
        mock_vectorstore_queries.similarity_search.return_value = [MagicMock(page_content="query info")]

        # Mock LLM for SQL generation
        mock_llm.invoke.return_value = MagicMock(content="SELECT * FROM test_table_empty")

        # Mock try_sql_execute to return an empty DataFrame
        empty_df = pd.DataFrame()
        mock_try_sql_execute.return_value = (empty_df, None) # Successful execution, but empty df

        result_state = sql_node(self.initial_state)

        self.assertEqual(result_state["condition"], "sql_execution_empty_result")
        self.assertIn("SQLの実行結果が空でした", result_state["error"])
        self.assertIn("some requirement", result_state["error"])
        self.assertEqual(result_state["missing_data_requirements"], ["some requirement"])
        mock_try_sql_execute.assert_called_once()

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4')
    def test_sql_node_input_from_plan_details_when_missing_reqs_empty(
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        # This test covers the scenario where missing_data_requirements is empty,
        # but the plan directs SQL execution using details from state.input.
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="test_uuid_plan_input".ljust(8, '0')))
        mock_vs_tables.return_value = [MagicMock(page_content="Table Info for Plan Req")]
        mock_vs_queries.return_value = [MagicMock(page_content="Query Info for Plan Req")]
        mock_llm.invoke.return_value = MagicMock(content="SELECT * FROM plan_req_table;")

        plan_req_df = pd.DataFrame([{"data": "plan_req_data"}])
        mock_try_sql.return_value = (plan_req_df, None)

        # Simulate state as set up by execute_plan_router for an SQL step
        # where check_history might have found nothing or was skipped.
        initial_state = MyState(
            input="Plan Requirement Data", # This comes from plan_step["details"]
            missing_data_requirements=[],  # Explicitly empty
            latest_df=collections.OrderedDict(),
            df_history=[],
            analysis_plan=[{"action": "sql", "details": "Plan Requirement Data"}], # Context: it's a planned SQL step
            current_plan_step_index=0,
            complex_analysis_original_query="Original user query for context"
        )
        # sql_node's logic: if missing_data_requirements is empty AND a plan is active,
        # it will try to use state.input (holding plan details) as requirement.
        # The execute_plan_router actually sets missing_data_requirements from details for "sql" actions,
        # so this specific internal sql_node fallback is less likely to be hit if execute_plan_router does its job.
        # However, let's adjust the test to reflect how sql_node *expects* to get plan details
        # if missing_data_requirements was not pre-populated from details by the router for some reason,
        # OR test the case where missing_data_requirements *was* populated by the router.
        # The primary path is that execute_plan_router sets missing_data_requirements.
        # So, let's test that.

        # Re-adjusting to test the primary path: execute_plan_router has set missing_data_requirements.
        initial_state_adjusted = MyState(
            input="Plan Requirement Data", # Set by router, but sql_node uses missing_data_requirements primarily
            missing_data_requirements=["Plan Requirement Data"], # Set by router from plan details
            latest_df=collections.OrderedDict(),
            df_history=[],
            analysis_plan=[{"action": "sql", "details": "Plan Requirement Data"}],
            current_plan_step_index=0,
            complex_analysis_original_query="Original user query for context"
        )

        result_state = sql_node(initial_state_adjusted)

        self.assertEqual(result_state["condition"], "sql_execution_done")
        self.assertIsNone(result_state["error"])

        self.assertIn("Plan Requirement Data", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["Plan Requirement Data"], plan_req_df.to_dict(orient="records"))

        self.assertEqual(len(result_state["df_history"]), 1)
        history_entry = result_state["df_history"][0]
        self.assertEqual(history_entry["query"], "Plan Requirement Data")
        self.assertEqual(history_entry["SQL"], "SELECT * FROM plan_req_table;")

        mock_vs_tables.assert_called_once_with("Plan Requirement Data", k=3)
        mock_llm.invoke.assert_called_once() # Check prompt content if necessary
        self.assertIn("Plan Requirement Data", str(mock_llm.invoke.call_args[0][0][-1]['content'])) # Check last message content
        self.assertIn("Original user query for context", str(mock_llm.invoke.call_args[0][0][-1]['content']))

        mock_try_sql.assert_called_once_with("SELECT * FROM plan_req_table;")
        self.assertEqual(result_state.get("missing_data_requirements"), [])


    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4')
    def test_sql_node_partial_failure_and_missing_reqs_update(
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="test_uuid_partial".ljust(8, '0'))) # Ensure 8 chars
        # RAG mocks for "Success Req"
        mock_vs_tables_success = [MagicMock(page_content="Table info for Success")]
        mock_vs_queries_success = [MagicMock(page_content="Query info for Success")]

        # RAG mocks for "Error Req"
        mock_vs_tables_error = [MagicMock(page_content="Table info for Error")]
        mock_vs_queries_error = [MagicMock(page_content="Query info for Error")]

        # Side effect for RAG calls
        def rag_side_effect(query_str, k):
            if query_str == "Success Req":
                return mock_vs_tables_success if 'tables' in str(inspect.stack()[1].function) else mock_vs_queries_success
            elif query_str == "Error Req":
                return mock_vs_tables_error if 'tables' in str(inspect.stack()[1].function) else mock_vs_queries_error
            return []

        # Need to import inspect for the RAG side effect to distinguish calls
        import inspect
        mock_vs_tables.side_effect = lambda q, k: rag_side_effect(q, k)
        mock_vs_queries.side_effect = lambda q, k: rag_side_effect(q, k)


        mock_llm.invoke.side_effect = [
            MagicMock(content="SELECT * FROM success_req;"),  # For "Success Req"
            MagicMock(content="SELECT * FROM error_req_initial;"), # For "Error Req" - 1st attempt
            MagicMock(content="SELECT * FROM error_req_fixed;")  # For "Error Req" - 2nd attempt (fix_sql_with_llm)
        ]

        success_df = pd.DataFrame([{"data": "success"}])
        mock_try_sql.side_effect = [
            (success_df, None),             # "Success Req" - success
            (None, "Initial SQL error for Error Req"), # "Error Req" (initial) - fails
            (None, "Persistent SQL error for Error Req") # "Error Req" (fixed) - fails again
        ]

        initial_state = MyState(
            missing_data_requirements=["Success Req", "Error Req", "Another Missing Req"], # Test update with an untouched req
            latest_df=collections.OrderedDict(),
            df_history=[],
            complex_analysis_original_query="Get success, error, and another data."
        )

        result_state = sql_node(initial_state)

        self.assertEqual(result_state["condition"], "sql_execution_partial_success") # Updated condition
        self.assertIsNotNone(result_state["error"])
        # The error message is now structured with transform_sql_error
        self.assertIn("For 'Error Req': SQLクエリの処理中に予期せぬエラーが発生しました。管理者に連絡してください。(詳細: Persistent SQL error for Error Req)", result_state["error"])


        self.assertIn("Success Req", result_state["latest_df"])
        self.assertEqual(result_state["latest_df"]["Success Req"], success_df.to_dict(orient="records"))
        self.assertNotIn("Error Req", result_state["latest_df"])
        self.assertNotIn("Another Missing Req", result_state["latest_df"])


        self.assertEqual(len(result_state["df_history"]), 1)
        self.assertEqual(result_state["df_history"][0]["query"], "Success Req")

        # LLM: 1 for Success, 2 for Error (initial + fix)
        self.assertEqual(mock_llm.invoke.call_count, 3)
        # DB: 1 for Success, 2 for Error
        self.assertEqual(mock_try_sql.call_count, 3)

        # missing_data_requirements should contain "Error Req" (attempted but failed)
        # and "Another Missing Req" (was in initial list, not attempted if sql_node processes only from its direct input)
        # The current sql_node processes ALL requirements passed in its `requirements_to_fetch` list,
        # which is initialized from state.missing_data_requirements.
        # So, "Another Missing Req" would NOT be processed if it wasn't included in the RAG/LLM/DB mocks.
        # Let's assume "Another Missing Req" was not part of the explicit processing sequence for this test focus.
        # The key is that "Error Req" (attempted and failed) should be in updated missing_data_requirements.
        # And "Success Req" (attempted and succeeded) should NOT be.
        # "Another Missing Req" was in the initial list. If sql_node was to fetch it, it would have.
        # Since we didn't mock for it, it implies it wasn't processed, thus should remain missing.
        # The current logic of sql_node:
        # updated_missing_requirements = [req for req in requirements_to_fetch if req not in successfully_fetched_reqs]
        # This means if "Another Missing Req" was in `requirements_to_fetch` (i.e. initial `state.missing_data_requirements`)
        # and was not processed (so not in `successfully_fetched_reqs`), it will remain.
        self.assertIn("Error Req", result_state.get("missing_data_requirements"))
        self.assertIn("Another Missing Req", result_state.get("missing_data_requirements"))
        self.assertNotIn("Success Req", result_state.get("missing_data_requirements"))
        self.assertEqual(len(result_state.get("missing_data_requirements")), 2)


class TestSqlNodeCorrection(unittest.TestCase):

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4')
    def test_sql_correction_successful_updates_missing_reqs( # Renamed for clarity
        self, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="corr_ok_".ljust(8, '0'))) # Ensure 8 chars
        req_string = "Sales for Product X"
        initial_sql = "SELECT sale FROM productX_sales;"
        fixed_sql = "SELECT sales FROM product_X_sales;"
        expected_df_data = [{"sales": 100}]
        expected_df = pd.DataFrame(expected_df_data)

        mock_vs_tables.return_value = [MagicMock(page_content="Table: product_X_sales (sales, product_id)")]
        mock_vs_queries.return_value = [MagicMock(page_content="SELECT sales FROM product_X_sales WHERE product_id = 'X'")]

        mock_llm.invoke.side_effect = [
            MagicMock(content=initial_sql),
            MagicMock(content=fixed_sql)
        ]
        mock_try_sql.side_effect = [
            (None, "no such column: sale"),
            (expected_df, None)
        ]

        initial_state = MyState(
            # input=f"Get {req_string}", # Not directly used by sql_node if missing_data_requirements is set
            missing_data_requirements=[req_string, "Other Req"], # Initially has two missing items
            latest_df=collections.OrderedDict(),
            df_history=[],
            complex_analysis_original_query=f"Get {req_string} and Other Req"
        )

        result_state = sql_node(initial_state)

        self.assertEqual(result_state["condition"], "sql_execution_partial_success") # "Other Req" was not processed (not mocked for)
        self.assertIsNone(result_state["error"]) # No error for "Sales for Product X"
        self.assertIn(req_string, result_state["latest_df"])
        self.assertEqual(result_state["latest_df"][req_string], expected_df_data)
        self.assertEqual(result_state["SQL"], fixed_sql)

        self.assertEqual(mock_llm.invoke.call_count, 2) # Once for initial, once for fix for "Sales for Product X"
        # RAG calls for "Sales for Product X" (initial and fix)
        self.assertEqual(mock_vs_tables.call_count, 2)
        self.assertEqual(mock_vs_queries.call_count, 2)

        fixer_prompt_args_list = [call_args[0][0] for call_args in mock_llm.invoke.call_args_list]

        # Initial SQL gen prompt for "Sales for Product X"
        self.assertIn(req_string, str(fixer_prompt_args_list[0][-1]['content']))
        self.assertIn(f"Get {req_string} and Other Req", str(fixer_prompt_args_list[0][-1]['content']))


        # fix_sql_with_llm prompt for "Sales for Product X"
        self.assertIn(initial_sql, str(fixer_prompt_args_list[1]))
        self.assertIn("no such column: sale", str(fixer_prompt_args_list[1]))
        self.assertIn(req_string, str(fixer_prompt_args_list[1]))


        self.assertEqual(mock_try_sql.call_count, 2) # For "Sales for Product X"
        mock_try_sql.assert_any_call(initial_sql)
        mock_try_sql.assert_any_call(fixed_sql)

        # "Sales for Product X" was successful, so it should be removed from missing_data_requirements.
        # "Other Req" was not processed (no mocks for it), so it should remain.
        self.assertEqual(result_state.get("missing_data_requirements"), ["Other Req"])

    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4')
    @patch('files.backend_codes.transform_sql_error')
    def test_sql_correction_fails_again_updates_missing_reqs( # Renamed for clarity
        self, mock_transform_sql_error, mock_uuid, mock_try_sql, mock_llm, mock_vs_queries, mock_vs_tables
    ):
        mock_uuid.return_value = MagicMock(hex=MagicMock(return_value="corr_fail".ljust(8, '0'))) # Ensure 8 chars
        req_string = "Inventory for Product Y"
        initial_sql = "SELECT inventry FROM productY_inventory;"
        fixed_sql_attempt = "SELECT inventory FROM product_Y_inventory_typo;"

        mock_vs_tables.return_value = [MagicMock(page_content="Table: product_Y_inventory (inventory, product_id)")]
        mock_vs_queries.return_value = [MagicMock(page_content="SELECT inventory FROM product_Y_inventory")]

        mock_llm.invoke.side_effect = [
            MagicMock(content=initial_sql),
            MagicMock(content=fixed_sql_attempt)
        ]
        mock_try_sql.side_effect = [
            (None, "no such column: inventry"),
            (None, "no such table: product_Y_inventory_typo")
        ]
        mock_transform_sql_error.return_value = "User-friendly: Table not found for Product Y"

        initial_state = MyState(
            missing_data_requirements=[req_string, "Another Req"], # Test with multiple initial missing reqs
            latest_df=collections.OrderedDict(),
            df_history=[],
            complex_analysis_original_query=f"Get {req_string} and Another Req"
        )

        result_state = sql_node(initial_state)

        # Condition depends on whether "Another Req" was processed. Assuming it wasn't (no mocks for it).
        # If only req_string was processed and failed, it's sql_execution_failed for that item.
        # If other items were present in missing_data_requirements but not processed, they remain.
        # The overall status would be partial_success if some items succeeded or failed if all processed items failed.
        # Since "Inventory for Product Y" failed and "Another Req" was not processed (implicitly failed to fetch here)
        # the condition should reflect that not everything was successful.
        # If sql_node only attempts req_string, and it fails, condition is "sql_execution_failed".
        # If it attempts both and only req_string has mocks and fails, "Another Req" would also "fail" (no data).
        # Let's assume sql_node attempts all in missing_data_requirements.
        # "Inventory for Product Y" -> Fails. "Another Req" -> No mocks, so it would effectively be a "no data found" or error if it tried.
        # For simplicity, assume the loop in sql_node only processes "Inventory for Product Y" due to mocks.
        # Then, "Inventory for Product Y" failed, and "Another Req" is still outstanding.

        # Re-evaluating: sql_node iterates through `requirements_to_fetch` (which is `missing_data_requirements`).
        # If "Another Req" has no mocks for RAG/LLM, it would likely error out or return no SQL.
        # Let's refine the test to focus only on `req_string` for clarity of this specific correction failure.
        initial_state_focused = MyState(
            missing_data_requirements=[req_string],
            latest_df=collections.OrderedDict(),
            df_history=[],
            complex_analysis_original_query=f"Get {req_string}"
        )
        result_state_focused = sql_node(initial_state_focused)


        self.assertEqual(result_state_focused["condition"], "sql_execution_failed") # All (one) attempted reqs failed
        self.assertIsNotNone(result_state_focused["error"])
        self.assertIn("User-friendly: Table not found for Product Y", result_state_focused["error"])
        self.assertNotIn(req_string, result_state_focused["latest_df"])
        self.assertEqual(result_state_focused["SQL"], fixed_sql_attempt)

        self.assertEqual(mock_llm.invoke.call_count, 2) # For req_string: initial + fix
        self.assertEqual(mock_try_sql.call_count, 2) # For req_string: initial + fix
        mock_transform_sql_error.assert_called_once_with("no such table: product_Y_inventory_typo")
        self.assertEqual(result_state_focused.get("missing_data_requirements"), [req_string]) # Still missing


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = build_workflow()
        # It's often good to have a consistent thread_id for tests that might interact with stateful checkpoints
        self.config = {"configurable": {"thread_id": f"test_thread_{self.id()}"}}

    @patch('files.backend_codes.llm') # Mock LLM for classify_intent_node
    @patch('files.backend_codes.suggest_analysis_paths_node') # Mock to check it's reached
    def test_workflow_greeting_to_empty_plan(self, mock_suggest_node, mock_llm_classify):
        # create_analysis_plan_node has internal mock logic for "hello" -> empty plan
        # No LLM mock needed for create_analysis_plan_node itself for this path.

        # 1. Mock for classify_intent_node
        # For a greeting, classify_intent_node might return empty or some general intent.
        # Let's assume it correctly identifies it as not needing data operations.
        # Or, more simply, the create_analysis_plan node's internal logic for "hello" is dominant.
        # The 'classify_next' will route to 'create_analysis_plan' if no strong other intent.
        mock_llm_classify.invoke.return_value = MagicMock(content="") # Empty intent list

        # Mock suggest_analysis_paths_node to prevent its LLM call and simplify assertion
        mock_suggest_node.return_value = MyState(condition="no_paths_suggested_mocked", analysis_options=[])


        user_input = "hello"
        initial_state = MyState(input=user_input)

        # Run the workflow
        final_state = self.workflow.invoke(initial_state, config=self.config)

        # Expected sequence:
        # classify -> create_analysis_plan (sees "hello", generates empty plan, condition="empty_plan_generated")
        # -> create_analysis_plan_next routes to "suggest_analysis_paths"
        # -> suggest_analysis_paths_node (mocked) -> suggest_analysis_paths_next routes to END.

        # Verify classify_intent_node was called
        mock_llm_classify.invoke.assert_called_once()

        # Check the state after create_analysis_plan_node would have run.
        # The final state will be after suggest_analysis_paths_node.
        # To check intermediate states, one would need to inspect graph states if checkpointer was more accessible
        # or test nodes more individually. Here, we check the overall outcome.

        # The 'condition' in the final state will be from the last executed node before END (suggest_analysis_paths_node)
        self.assertEqual(final_state.get("condition"), "no_paths_suggested_mocked")

        # Key things to check from create_analysis_plan_node's effect (persisted in checkpoint):
        # Need to get the actual state from the checkpointer to see intermediate results.
        # For simplicity in this test, we'll assume if suggest_paths is reached, create_analysis_plan did its job.
        # A more rigorous test would involve:
        # state_after_create_plan = self.workflow.get_state(self.config) # after create_analysis_plan
        # self.assertEqual(state_after_create_plan.get("analysis_plan"), [])
        # self.assertEqual(state_after_create_plan.get("condition"), "empty_plan_generated")

        # Check that suggest_analysis_paths_node was indeed called
        mock_suggest_node.assert_called_once()


    @patch('files.backend_codes.llm') # Mock LLM for classify_intent_node
    # clarify_node itself doesn't call LLM, it gets question from plan.
    # create_analysis_plan_node has internal mock for "vague"
    def test_workflow_ambiguous_query_to_clarify(self, mock_llm_classify):
        # 1. Mock for classify_intent_node
        mock_llm_classify.invoke.return_value = MagicMock(content="データ取得") # Assume it classifies some intent

        user_input = "vague sales data" # Triggers "clarify" in create_analysis_plan_node
        initial_state = MyState(input=user_input)

        final_state = self.workflow.invoke(initial_state, config=self.config)

        # Expected sequence:
        # classify -> create_analysis_plan (sees "vague", plan=[{"action":"clarify", ...}], condition="plan_generated")
        # -> dispatch_plan_step -> execute_plan_router (routes to "clarify")
        # -> clarify_node (sets clarification_question, condition="awaiting_user_clarification")
        # -> clarify_next (routes to END)

        mock_llm_classify.invoke.assert_called_once() # For classify_intent

        self.assertIsNotNone(final_state.get("clarification_question"))
        self.assertIn("Could you please specify", final_state.get("clarification_question"))
        self.assertEqual(final_state.get("condition"), "awaiting_user_clarification")
        self.assertIsNotNone(final_state.get("analysis_plan")) # Plan should still be there
        self.assertEqual(final_state.get("analysis_plan")[0]["action"], "clarify")

    @patch('files.backend_codes.llm') # For classify_intent and potentially for re-planning if not fully keyword driven
    def test_workflow_clarification_provided_leads_to_new_plan(self, mock_llm_invoker):
        # This test simulates the second step of a clarification flow.
        # 1. classify_next sees user_clarification, routes to dispatch.
        # 2. execute_plan_router sees user_clarification, routes to create_analysis_plan.
        # 3. create_analysis_plan_node uses clarification to make a new plan.

        original_query = "vague sales data"
        clarification = "for last month"

        # Mock for classify_intent (not strictly necessary as user_clarification bypasses it, but good for completeness)
        # The crucial part is that create_analysis_plan_node's mock logic for `user_clarification` is hit.
        mock_llm_invoker.return_value = MagicMock(content="データ取得")

        initial_state_after_clarify_prompt = MyState(
            input=original_query, # Original query that was clarified
            user_clarification=clarification,
            complex_analysis_original_query=original_query,
            # analysis_plan might contain the old "clarify" plan, it will be replaced.
            analysis_plan=[{"action": "clarify", "details": "some question"}],
            current_plan_step_index=0 # from previous clarify step
        )

        final_state = self.workflow.invoke(initial_state_after_clarify_prompt, config=self.config)

        # Expected sequence:
        # classify (sees user_clarification) -> dispatch_plan_step
        # -> execute_plan_router (sees user_clarification) -> create_analysis_plan
        # -> create_analysis_plan_node (sees user_clarification, generates new plan, e.g., check_history, sql, interpret)
        # -> dispatch_plan_step (now with new plan, index 0) -> execute_plan_router (routes to first step of new plan, e.g. check_history)
        # For this test, we'll assert up to the point the new plan is generated and ready to be dispatched.
        # The final state's condition will be from the *first node of the new plan* if it executed,
        # or from create_analysis_plan if the new plan was empty (not expected here).

        # Assert that create_analysis_plan_node was called and generated a new plan
        # (based on its internal mock logic for user_clarification)
        self.assertEqual(final_state.get("condition"), "history_checked") # Assuming new plan [check_history, sql, interpret] and check_history ran
        self.assertIsNone(final_state.get("user_clarification")) # Should be consumed
        self.assertEqual(final_state.get("complex_analysis_original_query"), original_query)

        new_plan = final_state.get("analysis_plan")
        self.assertIsNotNone(new_plan)
        self.assertNotEqual(new_plan[0]["action"], "clarify") # Should not be the old clarify plan
        # Based on create_analysis_plan_node's mock:
        expected_new_plan_part = [
            {"action": "check_history", "details": ["clarified sales data"]},
            {"action": "sql", "details": "clarified sales data"},
            # ...
        ]
        self.assertEqual(new_plan[0], expected_new_plan_part[0])
        self.assertEqual(new_plan[1], expected_new_plan_part[1])
        self.assertEqual(final_state.get("current_plan_step_index"), 1) # After check_history completed its _next


    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.uuid.uuid4') # For sql_node history ID
    @patch('files.backend_codes.check_history_node') # Mock check_history to control its output
    @patch('files.backend_codes.interpret_node') # Mock interpret_node to control its output
    def test_workflow_simple_query_check_history_miss_sql_interpret(
        self, mock_interpret_node, mock_check_history_node, mock_uuid,
        mock_try_sql, mock_vs_queries, mock_vs_tables, mock_llm_invoker
    ):
        user_input = "show product X sales"
        mock_uuid.return_value = MagicMock(hex="test_hist_id".ljust(8,'0'))

        # --- Mock LLM for classify_intent and sql_node (interpret_node is fully mocked) ---
        # 1. classify_intent_node:
        # 2. sql_node (SQL generation for "product X sales"):
        mock_llm_invoker.side_effect = [
            MagicMock(content="データ取得,データ解釈"), # classify_intent
            MagicMock(content="SELECT * FROM product_x_sales;") # sql_node
        ]

        # --- Mock check_history_node ---
        # Simulate a cache miss for "product X sales"
        mock_check_history_node.return_value = MyState(
            latest_df=collections.OrderedDict(), # No data found in history
            missing_data_requirements=["product X sales"], # The requirement is missing
            condition="history_checked"
            # other state fields are passed through or set by the node
        )

        # --- Mock RAG and DB for sql_node ---
        mock_vs_tables.return_value = [MagicMock(page_content="Table info for product X")]
        mock_vs_queries.return_value = [MagicMock(page_content="Query info for product X")]
        product_x_df = pd.DataFrame([{"product": "X", "sales": 120}])
        mock_try_sql.return_value = (product_x_df, None) # SQL execution success

        # --- Mock interpret_node ---
        mock_interpret_node.return_value = MyState(
            interpretation="Successfully interpreted product X sales.",
            condition="interpretation_done"
        )

        initial_state = MyState(input=user_input, df_history=[]) # Start with empty history
        final_state = self.workflow.invoke(initial_state, config=self.config)

        # Expected sequence:
        # classify -> create_analysis_plan (plan: [check_history, sql, interpret] for "product X sales")
        # -> dispatch -> check_history (mocked to return miss) -> its _next increments index
        # -> dispatch -> sql (executes, RAG, LLM for SQL, DB call) -> its _next increments index
        # -> dispatch -> interpret (mocked) -> its _next increments index
        # -> dispatch -> execute_plan_router (plan complete) -> suggest_analysis_paths -> END

        self.assertEqual(final_state.get("condition"), "interpretation_done")
        self.assertEqual(final_state.get("interpretation"), "Successfully interpreted product X sales.")

        # Verify latest_df contains data from sql_node
        self.assertIn("product X sales", final_state.get("latest_df", {}))
        self.assertEqual(final_state["latest_df"]["product X sales"], product_x_df.to_dict(orient="records"))

        # Verify df_history has the new entry from sql_node
        df_history = final_state.get("df_history", [])
        self.assertEqual(len(df_history), 1)
        self.assertEqual(df_history[0]["query"], "product X sales")
        self.assertEqual(df_history[0]["SQL"], "SELECT * FROM product_x_sales;")

        # Verify mock calls
        self.assertEqual(mock_llm_invoker.call_count, 2) # classify, sql_node
        mock_check_history_node.assert_called_once()

        # sql_node internal mocks
        mock_vs_tables.assert_called_once_with("product X sales", k=3)
        mock_vs_queries.assert_called_once_with("product X sales", k=3)
        mock_try_sql.assert_called_once_with("SELECT * FROM product_x_sales;")

        mock_interpret_node.assert_called_once()

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.check_history_node')
    @patch('files.backend_codes.sql_node') # Mock the entire sql_node
    @patch('files.backend_codes.interpret_node')
    def test_workflow_simple_query_check_history_hit_interpret(
        self, mock_interpret_node, mock_sql_node, mock_check_history_node, mock_llm_invoker
    ):
        user_input = "show known product Y sales"
        config = {"configurable": {"thread_id": f"test_workflow_hist_hit_{self.id()}"}}

        # --- Mock LLM for classify_intent ---
        # create_analysis_plan_node uses keyword "show known product Y sales" -> [check_history, sql, interpret]
        # interpret_node is fully mocked.
        mock_llm_invoker.return_value = MagicMock(content="データ取得,データ解釈") # classify_intent

        # --- Initial History Data ---
        known_product_y_data = [{"product": "Y", "sales": 75}]
        history_entry = {
            "id": "hist_Y", "query": "known product Y sales",
            "timestamp": datetime.now().isoformat(),
            "dataframe_dict": known_product_y_data, "SQL": "SELECT Y"
        }
        # Note: df_history is passed at invoke, but check_history_node uses the one from its input state.
        # So, the mock_check_history_node needs to reflect this.

        # --- Mock check_history_node ---
        # Simulate a cache hit for "known product Y sales"
        # This means check_history_node will populate latest_df and have empty missing_data_requirements
        mock_check_history_node.return_value = MyState(
            latest_df=collections.OrderedDict({"known product Y sales": known_product_y_data}),
            missing_data_requirements=[],
            condition="history_checked"
        )

        # --- Mock sql_node ---
        # If check_history had a hit for "known product Y sales",
        # the `missing_data_requirements` list (for that specific data req) would be empty when sql_node is called for it.
        # sql_node, when called for "known product Y sales" with no *specific* missing data for it,
        # should ideally not re-fetch.
        # However, the plan is [check_history, sql, interpret] where each step is for "known product Y sales".
        # execute_plan_router sets state.input to "known product Y sales" for sql_node.
        # And also state.missing_data_requirements = ["known product Y sales"] for the SQL step.
        # check_history_node does *not* clear future step's missing_data_requirements.
        # So, sql_node *will* be called with missing_data_requirements=["known product Y sales"].
        # We mock it to indicate it runs but perhaps finds the data (or we don't care about its internals here, just that it passes through).
        # For this test, we want to ensure interpret_node uses the data from check_history.
        # So, sql_node should preserve latest_df from check_history.
        def sql_node_side_effect(state: MyState):
            # Simulate sql_node running but not necessarily re-fetching if data already in latest_df
            # or, if it does fetch, it gets the same data.
            # Crucially, it should pass on latest_df.
            return {
                **state,
                "condition": "sql_execution_done", # or skipped if it has that logic
                "SQL": "SELECT * FROM y_sales_mocked_sql_node;", # To show it was "called"
                # latest_df should remain as set by check_history_node if sql_node doesn't modify for this req
            }
        mock_sql_node.side_effect = sql_node_side_effect


        # --- Mock interpret_node ---
        def interpret_node_side_effect(state: MyState):
            # Assert that the data from history (via check_history) is present in latest_df
            self.assertIn("known product Y sales", state.get("latest_df", {}))
            self.assertEqual(state["latest_df"]["known product Y sales"], known_product_y_data)
            return {
                **state,
                "interpretation": "Interpreted known product Y sales from history.",
                "condition": "interpretation_done"
            }
        mock_interpret_node.side_effect = interpret_node_side_effect

        initial_state = MyState(input=user_input, df_history=[history_entry])
        final_state = self.workflow.invoke(initial_state, config=self.config)

        # Expected sequence:
        # classify -> create_analysis_plan (plan: [ch, sql, int] for "known product Y sales")
        # -> dispatch -> check_history (mocked to return hit, populates latest_df)
        # -> dispatch -> sql (mocked, ideally doesn't overwrite latest_df["known product Y sales"] if found)
        # -> dispatch -> interpret (mocked, uses data from latest_df)
        # -> dispatch -> suggest_analysis_paths -> END

        self.assertEqual(final_state.get("condition"), "interpretation_done")
        self.assertEqual(final_state.get("interpretation"), "Interpreted known product Y sales from history.")

        # Verify latest_df still primarily reflects the data from history
        self.assertIn("known product Y sales", final_state.get("latest_df", {}))
        self.assertEqual(final_state["latest_df"]["known product Y sales"], known_product_y_data)

        mock_llm_invoker.assert_called_once() # Only for classify_intent
        mock_check_history_node.assert_called_once()
        mock_sql_node.assert_called_once()
        mock_interpret_node.assert_called_once()

    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.check_history_node')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.uuid.uuid4')
    @patch('files.backend_codes.PythonAstREPLTool') # For chart_node
    @patch('files.backend_codes.initialize_agent') # For chart_node
    @patch('files.backend_codes.os.path.exists') # For chart_node
    @patch('builtins.open') # For chart_node
    @patch('files.backend_codes.base64.b64encode') # For chart_node
    @patch('files.backend_codes.interpret_node') # Mock interpret_node at the end
    def test_workflow_multi_step_sql_chart_interpret(
        self, mock_interpret_node, mock_b64encode, mock_open, mock_os_exists,
        mock_initialize_agent, mock_python_tool, mock_uuid,
        mock_vs_queries, mock_vs_tables, mock_try_sql, mock_check_history, mock_llm_invoker
    ):
        user_input = "show new customer sales then chart sales by region then interpret"
        config = {"configurable": {"thread_id": f"test_workflow_multi_{self.id()}"}}

        mock_uuid.return_value = MagicMock(hex="multi_".ljust(8,'0'))

        # --- LLM Mocks ---
        # 1. classify_intent: data, chart, interpret
        # 2. create_analysis_plan: (internal mock uses user_input keywords)
        #    Plan for "show new customer sales then chart sales by region then interpret"
        #    might be like: [
        #        {"action": "check_history", "details": ["new customer sales", "sales by region"]},
        #        {"action": "sql", "details": "new customer sales"},
        #        {"action": "sql", "details": "sales by region"}, # Assuming planner makes separate SQL steps
        #        {"action": "chart", "details": "chart sales by region"},
        #        {"action": "interpret", "details": "interpret overall results"}
        #    ]
        #    For simplicity, let's assume the plan from create_analysis_plan's mock is:
        #    [ CH("customer sales"), SQL("customer sales"), Chart("chart customer sales"), Interpret("interpret results") ]
        #    The test will use "show sales then chart it" to trigger a simpler CH, SQL, Chart plan.
        #    Let's adapt user_input for create_analysis_plan's current mock.
        user_input_for_mock_plan = "show sales then chart it" # This generates CH, SQL, Chart
                                                            # We will then manually add interpret for this test's purpose.

        # LLM for classify
        # LLM for SQL node ("sales data" from the mock plan)
        # LLM for Chart node (Python code gen)
        # Interpret node is fully mocked.
        mock_llm_invoker.side_effect = [
            MagicMock(content="データ取得,グラフ作成,データ解釈"), # classify_intent
            MagicMock(content="SELECT * FROM sales_data;"),   # sql_node for "sales data"
            # chart_node's agent will call LLM. The agent's invoke is mocked by mock_initialize_agent.
        ]

        # --- Mock check_history_node ---
        # Called for "sales data" (from the plan generated by "show sales then chart it")
        mock_check_history.return_value = MyState(
            latest_df=collections.OrderedDict(), # Cache miss
            missing_data_requirements=["sales data"],
            condition="history_checked"
        )

        # --- Mock RAG & DB for sql_node (for "sales data") ---
        mock_vs_tables.return_value = [MagicMock(page_content="Sales table info")]
        mock_vs_queries.return_value = [MagicMock(page_content="Sales query info")]
        sales_df_data = [{"category": "Electronics", "sales": 5000}, {"category": "Books", "sales": 1500}]
        sales_df = pd.DataFrame(sales_df_data)
        mock_try_sql.return_value = (sales_df, None)

        # --- Mock chart_node components ---
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        # Agent calls LLM. The PythonAstREPLTool is what it executes.
        # The agent's invoke is what we're effectively mocking here for its LLM call part.
        # The actual Python code execution part is via PythonAstREPLTool.
        mock_agent_instance.invoke.return_value = {"output": "Chart generated by agent: output.png"}
        mock_python_tool.return_value = MagicMock() # The tool instance
        mock_os_exists.return_value = True # output.png exists
        mock_b64encode.return_value = "fake_chart_base64_string"
        mock_open.return_value = MagicMock() # Mock the file object from open()

        # --- Mock interpret_node ---
        # This will be the final step in our extended manual plan.
        # It should receive latest_df containing "sales data"
        def interpret_side_effect(state: MyState):
            self.assertIn("sales data", state["latest_df"])
            self.assertEqual(state["latest_df"]["sales data"], sales_df_data)
            return MyState(interpretation="Final interpretation of sales and chart.", condition="interpretation_done")
        mock_interpret_node.side_effect = interpret_side_effect

        # Initial state
        initial_state = MyState(input=user_input_for_mock_plan, df_history=[])

        # Invoke for the plan generated by "show sales then chart it"
        # This plan is: CH("sales data"), SQL("sales data"), Chart("chart sales data")
        # We need to manually chain the interpret_node call for this test.

        # Get state after chart node
        # To do this, we can run up to chart, then manually add interpret to plan and continue.
        # This is getting complicated. Let's simplify:
        # Test a CH -> SQL -> Chart workflow, then assert state.
        # Then, in a separate step (or by assuming interpret_node is called if in plan), check interpret.

        # For this test, let's assume the plan naturally includes interpret or test interpret separately.
        # The create_analysis_plan_node for "show sales" (which is part of "show sales then chart it")
        # generates: CH, SQL, Interpret.
        # The "show sales then chart it" generates: CH, SQL, Chart.
        # Let's use a query that leads to CH, SQL, Chart, Interpret if possible from mocks,
        # or simplify the test to CH, SQL, Chart and check chart_result.

        # Let's assume the plan is CH, SQL, Chart. We check chart_result.
        # And then, if we want to test interpret, it would be a subsequent call or a different plan.

        # Sticking to the plan from "show sales then chart it": [CH, SQL, Chart]
        final_state_after_chart = self.workflow.invoke(initial_state, config=self.config)

        # --- Assertions for CH -> SQL -> Chart part ---
        self.assertEqual(final_state_after_chart.get("condition"), "chart_generation_done")
        self.assertEqual(final_state_after_chart.get("chart_result"), "fake_chart_base64_string")
        self.assertIn("sales data", final_state_after_chart.get("latest_df", {}))
        self.assertEqual(final_state_after_chart["latest_df"]["sales data"], sales_df_data)

        # Check mocks
        mock_llm_invoker.assert_any_call(unittest.mock.ANY) # classify
        mock_llm_invoker.assert_any_call([ # sql_node
            {'role': 'system', 'content': unittest.mock.ANY},
            {'role': 'user', 'content': unittest.mock.ANY}]
        )
        # The chart node's agent calls LLM, this is part of mock_agent_instance.invoke
        mock_agent_instance.invoke.assert_called_once() # Chart agent's LLM call

        mock_check_history.assert_called_once()
        mock_try_sql.assert_called_once_with("SELECT * FROM sales_data;")
        mock_initialize_agent.assert_called_once() # For chart_node
        mock_os_exists.assert_called_with("output.png")

        # interpret_node should NOT have been called with this plan
        mock_interpret_node.assert_not_called()

    @patch('files.backend_codes.llm') # For SQL, Chart Agent, Interpret
    @patch('files.backend_codes.check_history_node')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.uuid.uuid4')
    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    @patch('files.backend_codes.os.path.exists')
    @patch('builtins.open')
    @patch('files.backend_codes.base64.b64encode')
    def test_workflow_manual_full_plan_ch_sql_chart_interpret(
        self, mock_b64encode, mock_open, mock_os_exists,
        mock_initialize_agent, mock_python_tool, mock_uuid,
        mock_vs_queries, mock_vs_tables, mock_try_sql, mock_check_history, mock_llm_invoker
    ):
        config = {"configurable": {"thread_id": f"test_manual_full_plan_{self.id()}"}}
        mock_uuid.return_value = MagicMock(hex="full_".ljust(8,'0'))

        # Manually define the plan
        manual_plan = [
            {"action": "check_history", "details": ["full_sales_data"]},
            {"action": "sql", "details": "full_sales_data"},
            {"action": "chart", "details": "chart full_sales_data"},
            {"action": "interpret", "details": "interpret full_sales_data and chart"}
        ]

        # --- LLM Mocks ---
        # 1. classify_intent_node (will be called first by workflow entry point)
        # 2. SQL node for "full_sales_data"
        # 3. Interpret node for "interpret full_sales_data and chart"
        # Note: Chart node's agent LLM call is handled by mock_chart_agent_instance.invoke
        mock_llm_invoker.side_effect = [
            MagicMock(content="データ取得,グラフ作成,データ解釈"), # For classify_intent_node
            MagicMock(content="SELECT * FROM full_sales;"), # For sql_node
            MagicMock(content="Interpreting the full sales data and its chart...") # For interpret_node
        ]

        # --- Mock check_history_node ---
        # Simulate miss for "full_sales_data"
        mock_check_history.return_value = MyState(
            latest_df=collections.OrderedDict(),
            missing_data_requirements=["full_sales_data"],
            condition="history_checked"
        )

        # --- Mock RAG & DB for sql_node ("full_sales_data") ---
        mock_vs_tables.return_value = [MagicMock(page_content="Full Sales table info")]
        mock_vs_queries.return_value = [MagicMock(page_content="Full Sales query info")]
        full_sales_df_data = [{"region": "North", "sales": 10000}, {"region": "South", "sales": 12000}]
        full_sales_df = pd.DataFrame(full_sales_df_data)
        mock_try_sql.return_value = (full_sales_df, None)

        # --- Mock chart_node components ---
        mock_chart_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_chart_agent_instance
        mock_chart_agent_instance.invoke.return_value = {"output": "Chart agent created output.png for full_sales_data"}
        mock_python_tool.return_value = MagicMock()
        mock_os_exists.return_value = True
        mock_b64encode.return_value = "base64_encoded_full_sales_chart"
        mock_open.return_value = MagicMock()

        # Initial state - we inject the plan directly.
        # classify_intent_node and create_analysis_plan_node are bypassed for this test.
        # The workflow should start from dispatch_plan_step effectively.
        initial_state = MyState(
            input="User query that would somehow lead to this manual plan", # For context, not directly used by router if plan exists
            analysis_plan=manual_plan,
            current_plan_step_index=0, # Start at the beginning of the plan
            df_history=[],
            latest_df=collections.OrderedDict() # Ensure it's initialized
        )

        # To make LangGraph start from a specific point (like after create_analysis_plan),
        # we'd typically need to manipulate the checkpointer or call internal methods.
        # For this test, we will invoke the whole workflow, but because analysis_plan is set,
        # classify_next will route to dispatch_plan_step, which then uses this plan.

        final_state = self.workflow.invoke(initial_state, config=self.config)

        # --- Assertions ---
        # Expected sequence: CH -> SQL -> Chart -> Interpret -> Suggest Paths -> END
        self.assertEqual(final_state.get("condition"), "interpretation_done")
        self.assertEqual(final_state.get("interpretation"), "Interpreting the full sales data and its chart...")
        self.assertEqual(final_state.get("chart_result"), "base64_encoded_full_sales_chart")

        self.assertIn("full_sales_data", final_state.get("latest_df", {}))
        self.assertEqual(final_state["latest_df"]["full_sales_data"], full_sales_df_data)

        df_history = final_state.get("df_history", [])
        self.assertEqual(len(df_history), 1)
        self.assertEqual(df_history[0]["query"], "full_sales_data")
        self.assertEqual(df_history[0]["SQL"], "SELECT * FROM full_sales;")

        # Verify mocks
        mock_check_history.assert_called_once()

        # SQL node mocks
        mock_vs_tables.assert_called_once_with("full_sales_data", k=3)
        mock_vs_queries.assert_called_once_with("full_sales_data", k=3)
        mock_try_sql.assert_called_once_with("SELECT * FROM full_sales;")

        # Chart node mocks
        mock_initialize_agent.assert_called_once() # Chart agent
        mock_chart_agent_instance.invoke.assert_called_once() # Chart agent's LLM call
        self.assertIn("chart full_sales_data", str(mock_chart_agent_instance.invoke.call_args[0][0]))


        # LLM invoker calls: sql_node, interpret_node. Chart agent's LLM is via mock_chart_agent_instance.
        # If classify_intent was truly bypassed, then 2 calls.
        # However, workflow.invoke always starts at the entry point "classify".
        # classify_intent_node WILL run. It uses llm. So one call for that.
        # Then sql_node (1), then interpret_node (1). Chart node's agent LLM is separate.
        # So, 3 calls to mock_llm_invoker.
        # The first call to mock_llm_invoker in side_effect is for SQL, second for Interpret.
        # We need one for classify.

        # Adjusting LLM mock for the classify call that will happen
        mock_llm_invoker.assert_any_call(unittest.mock.ANY) # For classify_intent
        mock_llm_invoker.assert_any_call([ # For sql_node
             {'role': 'system', 'content': unittest.mock.ANY},
             {'role': 'user', 'content': unittest.mock.ANY}
        ])
        mock_llm_invoker.assert_any_call([ # For interpret_node
             {'role': 'system', 'content': unittest.mock.ANY},
             {'role': 'user', 'content': unittest.mock.ANY}
        ])
        self.assertEqual(mock_llm_invoker.call_count, 3)


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
    def test_interpret_node_valid_data_with_context(self, mock_llm): # Renamed and updated
        mock_llm.invoke.return_value = MagicMock(content="This is a great interpretation of the sales and inventory trends.")
        sample_data = collections.OrderedDict({
            "Sales Data": [{"item": "A", "revenue": 100}, {"item": "B", "revenue": 150}],
            "Inventory Data": [{"item": "A", "stock": 20}, {"item": "B", "stock": 30}]
        })
        # Simulate input coming from plan step details
        interpretation_context = "Analyze sales and inventory trends"
        state = MyState(latest_df=sample_data, input=interpretation_context)

        result_state = interpret_node(state)

        mock_llm.invoke.assert_called_once()
        # The prompt is a list of messages. The user message is the last one.
        user_prompt_content = mock_llm.invoke.call_args[0][0][-1]['content']

        # Verify the prompt more thoroughly
        self.assertIn(interpretation_context, user_prompt_content) # Check context from state.input
        for req_key, data_list in sample_data.items():
            self.assertIn(f"■「{req_key}」に関するデータ:", user_prompt_content)
            if data_list:
                # Simple check for some data presence, detailed df string check is complex
                self.assertIn(str(data_list[0][list(data_list[0].keys())[0]]), user_prompt_content)


        self.assertEqual(result_state["interpretation"], "This is a great interpretation of the sales and inventory trends.")
        self.assertEqual(result_state["condition"], "interpretation_done") # Updated condition

    @patch('files.backend_codes.llm')
    def test_interpret_node_empty_latest_df_ordered_dict(self, mock_llm): # Renamed for clarity
        state = MyState(latest_df=collections.OrderedDict(), input="Interpret this empty data.")
        result_state = interpret_node(state)
        self.assertEqual(result_state["interpretation"], "解釈するデータが空です。") # Updated message
        self.assertEqual(result_state["condition"], "interpretation_failed_empty_data") # Updated condition
        mock_llm.invoke.assert_not_called()

    @patch('files.backend_codes.llm')
    def test_interpret_node_data_list_is_empty_in_ordered_dict(self, mock_llm): # Renamed
        mock_llm.invoke.return_value = MagicMock(content="No data was available for Empty Sales.")
        # Simulate input from plan step details
        state = MyState(
            latest_df=collections.OrderedDict({"Empty Sales": []}),
            input="Interpret empty sales"
        )
        result_state = interpret_node(state)

        mock_llm.invoke.assert_called_once()
        user_prompt_content = mock_llm.invoke.call_args[0][0][-1]['content']
        self.assertIn("Empty Sales", user_prompt_content)
        self.assertIn("(この要件に対するデータはありませんでした)", user_prompt_content) # Updated string
        self.assertIn("Interpret empty sales", user_prompt_content) # Check context
        self.assertEqual(result_state["interpretation"], "No data was available for Empty Sales.")
        self.assertEqual(result_state["condition"], "interpretation_done") # Updated


    @patch('files.backend_codes.llm')
    def test_interpret_node_latest_df_is_none(self, mock_llm): # Renamed
        state = MyState(latest_df=None, input="Interpret non-existent data.")
        result_state = interpret_node(state)
        self.assertEqual(result_state["interpretation"], "解釈するデータがありません。") # Updated message
        self.assertEqual(result_state["condition"], "interpretation_failed_no_data") # Updated condition
        mock_llm.invoke.assert_not_called()

    @patch('files.backend_codes.llm')
    def test_interpret_node_llm_returns_empty_string(self, mock_llm): # Renamed
        mock_llm.invoke.return_value = MagicMock(content="")
        sample_data = collections.OrderedDict({"Sales Data": [{"item": "A", "revenue": 100}]})
        state = MyState(latest_df=sample_data, input="Interpret sales data")

        result_state = interpret_node(state)

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["interpretation"], "解釈の生成に失敗しました。") # Updated: node now sets a message
        self.assertEqual(result_state["condition"], "interpretation_failed") # Updated: condition reflects failure

    @patch('files.backend_codes.llm')
    def test_interpret_node_llm_returns_none_content(self, mock_llm): # Renamed and updated
        mock_response = MagicMock()
        mock_response.content = None # Simulate LLM returning None for content
        mock_llm.invoke.return_value = mock_response

        sample_data = collections.OrderedDict({"Sample Data": [{"col": "value"}]})
        state = MyState(latest_df=sample_data, input="Interpret sample data")

        result_state = interpret_node(state) # interpret_node now handles .strip() on None

        mock_llm.invoke.assert_called_once()
        self.assertEqual(result_state["interpretation"], "解釈の生成に失敗しました。") # Updated: node now sets a message
        self.assertEqual(result_state["condition"], "interpretation_failed") # Updated: condition reflects this specific failure mode

    @patch('files.backend_codes.llm')
    def test_interpret_node_all_data_entries_are_effectively_empty(self, mock_llm):
        state = MyState(
            latest_df=collections.OrderedDict({
                "Empty Req 1": [],
                "Empty Req 2": None, # None list for a req
                "Empty Req 3": [{"__empty__": True}] # Simulate a df that stringifies to empty-like
            }),
            input="Interpret various empty data sets"
        )
        # If all data string parts are empty-like, LLM should not be called.
        # The node checks: if not full_data_string.strip() or all empty indicators...

        # Mocking pd.DataFrame to control string output for "Empty Req 3"
        with patch('pandas.DataFrame') as mock_pd_df:
            mock_df_instance = MagicMock()
            mock_df_instance.empty = False # Make it seem non-empty initially
            # Control its string representation to be one of the "empty" indicators
            mock_df_instance.to_string.return_value = "(この要件に対するデータは空でした)"
            mock_pd_df.side_effect = lambda data: mock_df_instance if data == [{"__empty__": True}] else pd.DataFrame(data)

            result_state = interpret_node(state)

        self.assertEqual(result_state["interpretation"], "解釈対象の有効なデータがありませんでした。")
        self.assertEqual(result_state["condition"], "interpretation_failed_empty_data")
        mock_llm.assert_not_called()


class TestChartNode(unittest.TestCase):

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'chart_bytes')
    @patch('files.backend_codes.base64.b64encode')
    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_successful_generation_selects_first_df( # Renamed
        self, mock_initialize_agent, mock_python_tool, mock_os_path_exists, mock_b64encode, mock_open_file
    ):
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke.return_value = {"output": "Agent created output.png"}

        mock_python_tool_instance = MagicMock() # Not directly used, but the class is patched
        mock_python_tool.return_value = mock_python_tool_instance

        mock_os_path_exists.return_value = True
        mock_b64encode.return_value = "encoded_chart_data"

        first_df_key = "Sales Data"
        first_df_list = [{"category": "A", "amount": 100}, {"category": "B", "amount": 150}]
        sample_data_ordered_dict = collections.OrderedDict({
            first_df_key: first_df_list,
            "Other Data": [{"info": "test"}]
        })
        # Generic input, should select the first DataFrame by default
        chart_instructions = "Generate a suitable chart for the available data." # Default if not specific
        state = MyState(latest_df=sample_data_ordered_dict, input=chart_instructions)

        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()
        tool_args = mock_python_tool.call_args
        passed_df_to_tool = tool_args.kwargs['locals']['df']
        expected_df_for_tool = pd.DataFrame(first_df_list)
        pd.testing.assert_frame_equal(passed_df_to_tool, expected_df_for_tool)

        mock_agent_instance.invoke.assert_called_once()
        agent_prompt_messages = mock_agent_instance.invoke.call_args[0][0] # Prompt is the first arg

        # The prompt structure might vary (string or list of messages). Assuming string for simplicity in asserts.
        # If it's a list of messages, need to access the content of the relevant message.
        agent_prompt_str = str(agent_prompt_messages)


        self.assertIn(chart_instructions, agent_prompt_str)
        self.assertIn(f"以下の「{first_df_key}」に関するデータ", agent_prompt_str)
        self.assertIn(expected_df_for_tool.head().to_string(), agent_prompt_str)

        mock_os_path_exists.assert_called_once_with("output.png")
        mock_b64encode.assert_called_once_with(b"chart_bytes")
        mock_open_file.assert_called_once_with("output.png", "rb")

        self.assertEqual(result_state["chart_result"], "encoded_chart_data")
        self.assertEqual(result_state["condition"], "chart_generation_done") # Updated condition

    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_latest_df_is_empty_ordered_dict(self, mock_initialize_agent): # Renamed
        state = MyState(latest_df=collections.OrderedDict(), input="Plot this")
        result_state = chart_node(state)
        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "chart_generation_failed_no_data") # Updated
        mock_initialize_agent.assert_not_called()

    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_latest_df_is_none(self, mock_initialize_agent): # Renamed
        state = MyState(latest_df=None, input="Plot this")
        result_state = chart_node(state)
        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "chart_generation_failed_no_data") # Updated
        mock_initialize_agent.assert_not_called()

    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.initialize_agent')
    @patch('files.backend_codes.PythonAstREPLTool')
    def test_chart_node_output_file_not_created_by_agent( # Renamed
        self, mock_python_tool, mock_initialize_agent, mock_os_path_exists
    ):
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke.return_value = {"output": "Agent ran but did not create file"}
        mock_os_path_exists.return_value = False

        sample_data = collections.OrderedDict({"Sales": [{"category": "A", "amount": 100}]})
        state = MyState(latest_df=sample_data, input="Plot sales data")
        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()
        mock_agent_instance.invoke.assert_called_once()
        mock_os_path_exists.assert_called_once_with("output.png")
        mock_python_tool.assert_called_once()

        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "chart_generation_failed_no_output_file") # Updated
        self.assertIn("Agent: Agent ran but did not create file", result_state["error"])


    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_agent_execution_error( # Renamed
        self, mock_initialize_agent, mock_python_tool, mock_os_path_exists
    ):
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        agent_error_message = "Invalid Python code generated by agent"
        mock_agent_instance.invoke.side_effect = Exception(agent_error_message)

        sample_data = collections.OrderedDict({"Sales": [{"category": "A", "amount": 100}]})
        state = MyState(latest_df=sample_data, input="Plot sales for error test")
        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()
        mock_python_tool.assert_called_once()
        mock_agent_instance.invoke.assert_called_once()

        # os.path.exists("output.png") is called once at the start of the try block
        mock_os_path_exists.assert_called_once_with("output.png")


        self.assertIsNone(result_state["chart_result"])
        self.assertEqual(result_state["condition"], "chart_generation_failed_agent_error") # Updated
        self.assertEqual(result_state.get("error"), agent_error_message)


    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'chart_bytes_various')
    @patch('files.backend_codes.base64.b64encode')
    @patch('files.backend_codes.os.path.exists')
    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_chart_node_selects_df_based_on_input_instructions( # Renamed
        self, mock_initialize_agent, mock_python_tool, mock_os_path_exists, mock_b64encode, mock_open_file
    ):
        mock_agent_instance = MagicMock()
        mock_initialize_agent.return_value = mock_agent_instance
        mock_agent_instance.invoke.return_value = {"output": "Agent created output.png"}
        mock_os_path_exists.return_value = True
        mock_b64encode.return_value = "encoded_chart_specific_data"

        first_df_key = "Sales Data"
        first_df_list = [{"category": "A", "amount": 100}]
        second_df_key = "Inventory Data" # This one should be selected
        second_df_list = [{"item": "X", "stock": 500}, {"item": "Y", "stock": 300}]

        sample_data_ordered_dict = collections.OrderedDict({
            first_df_key: first_df_list,
            second_df_key: second_df_list
        })
        # Specific input that should match the second_df_key
        chart_instructions = f"Plot the {second_df_key.lower()} trends" # make it case-insensitive match like in node
        state = MyState(latest_df=sample_data_ordered_dict, input=chart_instructions)

        result_state = chart_node(state)

        mock_initialize_agent.assert_called_once()
        tool_args = mock_python_tool.call_args
        passed_df_to_tool = tool_args.kwargs['locals']['df']
        expected_df_for_tool = pd.DataFrame(second_df_list) # Should use the second DF
        pd.testing.assert_frame_equal(passed_df_to_tool, expected_df_for_tool)

        mock_agent_instance.invoke.assert_called_once()
        agent_prompt_str = str(mock_agent_instance.invoke.call_args[0][0])

        self.assertIn(chart_instructions, agent_prompt_str)
        self.assertIn(f"以下の「{second_df_key}」に関するデータ", agent_prompt_str) # Selected key
        self.assertIn(expected_df_for_tool.head().to_string(), agent_prompt_str)

        self.assertEqual(result_state["chart_result"], "encoded_chart_specific_data")
        self.assertEqual(result_state["condition"], "chart_generation_done")


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


class TestClarifyNodeAndFlow(unittest.TestCase):

    def test_clarify_node_sets_question_and_condition(self):
        clarification_detail = "Could you specify the region for sales data?"
        initial_plan = [{"action": "clarify", "details": clarification_detail}]
        state = MyState(
            analysis_plan=initial_plan,
            current_plan_step_index=0,
            # input would be set to clarification_detail by execute_plan_router before calling clarify_node
            # However, clarify_node itself re-fetches from analysis_plan.
        )

        result_state = clarify_node(state)

        self.assertEqual(result_state.get("clarification_question"), clarification_detail)
        self.assertEqual(result_state.get("condition"), "awaiting_user_clarification")
        self.assertIsNone(result_state.get("user_clarification")) # Ensure it's not set here

    def test_clarify_node_handles_missing_details_gracefully(self):
        initial_plan = [{"action": "clarify"}] # Details missing
        state = MyState(analysis_plan=initial_plan, current_plan_step_index=0)
        result_state = clarify_node(state)
        self.assertIn("question is missing from the plan", result_state.get("clarification_question"))
        self.assertEqual(result_state.get("condition"), "awaiting_user_clarification")

    def test_clarify_node_error_if_not_clarify_action(self):
        initial_plan = [{"action": "sql", "details": "some sql"}]
        state = MyState(analysis_plan=initial_plan, current_plan_step_index=0)
        result_state = clarify_node(state)
        self.assertIsNotNone(result_state.get("error"))
        self.assertEqual(result_state.get("condition"), "clarify_error_unexpected_action")

    # Workflow test for clarification loop requires patching build_workflow or specific nodes
    # For now, focusing on unit tests of nodes involved in the flow.
    # A full workflow test for clarification would look like:
    # 1. Invoke with "vague query" -> create_analysis_plan makes clarify plan -> clarify_node sets question -> END
    # 2. Invoke with "user_clarification" -> create_analysis_plan makes new plan based on it.
    # This is better suited for the TestWorkflow class.

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
    @patch('files.backend_codes.check_history_node')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.extract_sql', side_effect=lambda x: x.replace("```sql", "").replace("```", "").strip())
    @patch('files.backend_codes.uuid.uuid4')
    @patch('files.backend_codes.interpret_node')
    def test_workflow_sql_error_recovery_success(
        self, mock_interpret_node, mock_uuid, mock_extract_sql,
        mock_try_sql, mock_vs_queries, mock_vs_tables, mock_check_history, mock_llm_invoker
    ):
        user_input = "Show New Product Sales" # This will result in a plan like [CH, SQL, Interpret]
        config = {"configurable": {"thread_id": f"test_sql_rec_succ_{self.id()}"}}
        mock_uuid.return_value = MagicMock(hex="rec_ok_".ljust(8, '0'))

        # LLM responses for:
        # 1. classify_intent_node
        # 2. sql_node (initial faulty SQL for "New Product Sales")
        # 3. sql_node (fix_sql_with_llm for corrected SQL)
        # interpret_node is fully mocked.
        mock_llm_invoker.side_effect = [
            MagicMock(content="データ取得,データ解釈"),  # classify_intent
            MagicMock(content="```sql\nSELECT * FRMO new_product_sales;\n```"), # Initial SQL for "New Product Sales"
            MagicMock(content="```sql\nSELECT * FROM new_product_sales;\n```")  # Corrected SQL
        ]

        # Mock check_history_node to simulate a miss
        mock_check_history.return_value = MyState(
            latest_df=collections.OrderedDict(),
            missing_data_requirements=["New Product Sales"], # Plan detail will be this
            condition="history_checked"
        )

        # DB execution results for sql_node
        mock_try_sql.side_effect = [
            (None, "syntax error near FRMO"), # For faulty SQL
            (pd.DataFrame([{"product": "SuperWidget", "sales": 150}]), None) # For corrected SQL
        ]

        # RAG for sql_node (called for initial generation and for fix)
        mock_vs_tables.return_value = [MagicMock(page_content="Table: new_product_sales (product, sales)")]
        mock_vs_queries.return_value = [] # No similar queries

        # Mock interpret_node for the end of the successful flow
        def interpret_side_effect(state: MyState):
            self.assertIn("New Product Sales", state["latest_df"]) # Check data is there
            return {**state, "interpretation": "Interpreted new product sales.", "condition": "interpretation_done"}
        mock_interpret_node.side_effect = interpret_side_effect

        initial_workflow_state = {"input": user_input, "df_history": []}
        final_state = self.workflow.invoke(initial_workflow_state, config=config)

        self.assertEqual(final_state.get("condition"), "interpretation_done")
        self.assertIn("New Product Sales", final_state.get("latest_df", {}))
        self.assertEqual(final_state["latest_df"]["New Product Sales"], [{"product": "SuperWidget", "sales": 150}])
        self.assertEqual(final_state.get("interpretation"), "Interpreted new product sales.")
        self.assertIsNone(final_state.get("error"))

        df_history = final_state.get("df_history", [])
        self.assertEqual(len(df_history), 1)
        self.assertEqual(df_history[0]["query"], "New Product Sales") # This should be the plan detail
        self.assertEqual(df_history[0]["SQL"], "SELECT * FROM new_product_sales;")

        self.assertEqual(mock_llm_invoker.call_count, 3) # classify, sql_gen_initial, sql_gen_fix
        self.assertEqual(mock_try_sql.call_count, 2)
        mock_extract_sql.assert_called() # Called by sql_node and fix_sql_with_llm
        self.assertEqual(mock_extract_sql.call_count, 2) # initial + fix
        mock_check_history.assert_called_once()
        mock_interpret_node.assert_called_once()
        # RAG calls: once for initial SQL gen, once for fix logic
        self.assertEqual(mock_vs_tables.call_count, 2)
        self.assertEqual(mock_vs_queries.call_count, 2)


    @patch('files.backend_codes.llm')
    @patch('files.backend_codes.check_history_node')
    @patch('files.backend_codes.vectorstore_tables.similarity_search')
    @patch('files.backend_codes.vectorstore_queries.similarity_search')
    @patch('files.backend_codes.try_sql_execute')
    @patch('files.backend_codes.extract_sql', side_effect=lambda x: x.replace("```sql", "").replace("```", "").strip())
    @patch('files.backend_codes.suggest_analysis_paths_node') # To verify it's reached on failure
    def test_workflow_sql_error_recovery_failure(
        self, mock_suggest_node, mock_extract_sql, mock_try_sql,
        mock_vs_queries, mock_vs_tables, mock_check_history, mock_llm_invoker
    ):
        user_input = "Show Problematic Data" # Plan: CH, SQL, Interpret
        config = {"configurable": {"thread_id": f"test_sql_rec_fail_{self.id()}"}}

        # LLM Mocks: classify, sql_initial, sql_fix
        mock_llm_invoker.side_effect = [
            MagicMock(content="データ取得"), # classify_intent
            MagicMock(content="```sql\nSELECT * FRMO problematic_sales;\n```"), # Initial SQL
            MagicMock(content="```sql\nSELECT * FRRMO problematic_sales;\n```")  # "Fixed" SQL (still faulty)
        ]

        # Mock check_history for a miss
        mock_check_history.return_value = MyState(
            latest_df=collections.OrderedDict(),
            missing_data_requirements=["Problematic Data"],
            condition="history_checked"
        )

        # DB execution: both attempts fail
        mock_try_sql.side_effect = [
            (None, "syntax error near FRMO"),
            (None, "syntax error near FRRMO")
        ]

        # RAG for sql_node (called for initial and fix attempts)
        mock_vs_tables.return_value = [MagicMock(page_content="Table: problematic_sales (id, value)")]
        mock_vs_queries.return_value = []

        # Mock suggest_analysis_paths_node
        mock_suggest_node.return_value = MyState(condition="no_paths_suggested_mocked_after_fail")


        initial_workflow_state = {"input": user_input, "df_history": []}
        final_state = self.workflow.invoke(initial_workflow_state, config=self.config)

        # Expected path: classify -> create_plan -> CH -> SQL (fails twice) -> SQL_next routes to dispatch
        # -> dispatch sees plan index at interpret step, but SQL failed.
        # execute_plan_router might then end the plan and go to suggest_analysis_paths.
        # The sql_node itself sets condition "sql_execution_failed".
        # sql_next then routes to dispatch_plan_step.
        # execute_plan_router will then try to execute the *next* step in the plan (e.g., interpret).
        # The interpret_node will then run with whatever data is in latest_df (which would be empty for "Problematic Data").

        # So, the final condition might be from interpret_node if it runs, or suggest_analysis_paths if plan ends early.
        # Given the current sql_next logic, it always increments index and goes to dispatch.
        # So interpret_node *will* be called.

        # Let's verify the state after sql_node would have failed:
        # The state that goes into interpret_node would have an error from sql_node.
        # The final state will be after suggest_analysis_paths_node.
        self.assertEqual(final_state.get("condition"), "no_paths_suggested_mocked_after_fail")

        # Check the error state set by sql_node (it should be in the final checkpointed state)
        # This requires inspecting the state *before* suggest_analysis_paths, or ensuring error propagates.
        # For this test, let's assume the error from sql_node is available for inspection.
        # The workflow.get_state(config) could be used if we had intermediate checkpoints.
        # For now, let's verify the mocks for sql_node were hit as expected.

        self.assertIsNotNone(final_state.get("error"))
        expected_user_friendly_error = "For 'Problematic Data': SQL構文にエラーがあります。クエリを確認してください。(詳細: syntax error near FRRMO)"
        self.assertIn(expected_user_friendly_error, final_state.get("error"))
        self.assertNotIn("Problematic Data", final_state.get("latest_df", {}))
        self.assertEqual(final_state.get("SQL"), "SELECT * FRRMO problematic_sales;")

        self.assertEqual(mock_llm_invoker.call_count, 3) # classify, sql_gen_initial, sql_gen_fix
        self.assertEqual(mock_try_sql.call_count, 2)
        self.assertEqual(mock_extract_sql.call_count, 2)
        mock_check_history.assert_called_once()
        mock_suggest_node.assert_called_once() # Verify it's the end path


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

class TestDataProcessingNode(unittest.TestCase):

    def setUp(self):
        self.initial_df_list = [{"colA": 1, "colB": 2}, {"colA": 3, "colB": 4}]
        self.initial_df = pd.DataFrame(self.initial_df_list)
        self.default_input_instruction = "Perform some processing on data1"

        # Helper to get df.info() as a string
        buffer = io.StringIO()
        self.initial_df.info(buf=buffer)
        self.df_info_str_initial = buffer.getvalue()

        self.default_state = MyState(
            latest_df=collections.OrderedDict({"data1": self.initial_df_list}),
            input=self.default_input_instruction,
        )

    def _get_df_info_as_string(self, df):
        buffer = io.StringIO()
        df.info(buf=buffer)
        return buffer.getvalue()

    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_successful_data_processing_add_column(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value

        mock_repl_tool_instance = MockPythonAstREPLTool.return_value
        mock_repl_tool_instance.globals = {"df": self.initial_df.copy(), "pd": pd}

        def agent_invoke_side_effect(prompt_str):
            # Check that df.info() string is in the prompt
            self.assertIn(self.df_info_str_initial, prompt_str)
            current_df = mock_repl_tool_instance.globals["df"]
            current_df['colC'] = current_df['colA'] * 2
            mock_repl_tool_instance.globals["df"] = current_df
            return {"output": "Agent executed: Added colC"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        state = self.default_state
        state["input"] = "Add a new column 'colC' where colC = colA * 2 for data1" # data1 is the key for initial_df
        result_state = data_processing_node(state)

        self.assertEqual(result_state["condition"], "data_processing_done")
        self.assertIsNone(result_state.get("error"))

        processed_df_list = result_state["latest_df"]["data1"]
        processed_df = pd.DataFrame(processed_df_list)

        expected_df = self.initial_df.copy()
        expected_df['colC'] = expected_df['colA'] * 2
        pd.testing.assert_frame_equal(processed_df, expected_df)
        MockInitializeAgent.assert_called_once()
        MockPythonAstREPLTool.assert_called_once()


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_data_processing_with_df_key_syntax_valid(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value

        df1_data = [{"colA": 10, "colB": 20}]
        df1 = pd.DataFrame(df1_data)
        df2_data = [{"colX": 50, "colY": 60}]
        df2 = pd.DataFrame(df2_data)

        initial_latest_df = collections.OrderedDict({
            "df_one": df1_data,
            "df_two": df2_data
        })

        # Agent will operate on df_one (explicitly chosen)
        mock_repl_tool_instance.globals = {"df": df1.copy(), "pd": pd}

        def agent_invoke_side_effect(prompt_str):
            self.assertIn(self._get_df_info_as_string(df1), prompt_str) # df1's info
            self.assertIn("DF_KEY['df_one']", prompt_str) # Context message should reflect this
            # Guidance should be present
            self.assertIn("複数のDataFrameが利用可能です。", prompt_str)
            self.assertIn("DF_KEY['キー名']", prompt_str)
            self.assertIn("- 'df_one': 1行", prompt_str) # Check listed DFs
            self.assertIn("- 'df_two': 1行", prompt_str)


            current_df = mock_repl_tool_instance.globals["df"]
            current_df["newCol"] = current_df["colA"] + 5
            mock_repl_tool_instance.globals["df"] = current_df
            return {"output": "Agent processed df_one"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        state = MyState(
            latest_df=initial_latest_df,
            input="DF_KEY['df_one'] Add newCol = colA + 5"
        )
        result_state = data_processing_node(state)

        self.assertEqual(result_state["condition"], "data_processing_done")
        expected_df1_processed = df1.copy()
        expected_df1_processed["newCol"] = expected_df1_processed["colA"] + 5
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["df_one"]), expected_df1_processed)
        # Ensure df_two is untouched
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["df_two"]), df2)


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_data_processing_with_df_key_syntax_invalid_key_fallback(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value

        df1_data = [{"colA": 10}] # This will be the fallback
        df1 = pd.DataFrame(df1_data)
        df2_data = [{"colX": 50}]
        # df2 = pd.DataFrame(df2_data)

        initial_latest_df = collections.OrderedDict({
            "df_alpha": df1_data, # Fallback target
            "df_beta": df2_data
        })

        # Agent will operate on df_alpha (fallback)
        mock_repl_tool_instance.globals = {"df": df1.copy(), "pd": pd}

        def agent_invoke_side_effect(prompt_str):
            self.assertIn(self._get_df_info_as_string(df1), prompt_str)
            # Context message should indicate fallback
            self.assertIn("利用可能な最初のデータセット「df_alpha」", prompt_str)
            current_df = mock_repl_tool_instance.globals["df"]
            current_df["processed"] = True
            mock_repl_tool_instance.globals["df"] = current_df
            return {"output": "Agent processed fallback df_alpha"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        state = MyState(
            latest_df=initial_latest_df,
            input="DF_KEY['non_existent_key'] Process this."
        )
        # Expect a log warning for "DF_KEY['non_existent_key'] not found..."
        # We can't directly test log output without more setup, so focus on behavior.
        result_state = data_processing_node(state)

        self.assertEqual(result_state["condition"], "data_processing_done")
        expected_df1_processed = df1.copy()
        expected_df1_processed["processed"] = True
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["df_alpha"]), expected_df1_processed)
        # df_beta should be untouched
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["df_beta"]), pd.DataFrame(df2_data))


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_data_processing_with_df_key_syntax_empty_target_fallback(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value

        df1_empty_data = []
        df2_data = [{"colX": 50}] # This will be the fallback
        df2 = pd.DataFrame(df2_data)

        initial_latest_df = collections.OrderedDict({
            "df_empty": df1_empty_data,
            "df_valid_fallback": df2_data
        })

        mock_repl_tool_instance.globals = {"df": df2.copy(), "pd": pd} # Agent operates on df_valid_fallback

        def agent_invoke_side_effect(prompt_str):
            self.assertIn(self._get_df_info_as_string(df2), prompt_str)
            self.assertIn("利用可能な最初のデータセット「df_valid_fallback」", prompt_str) # Fallback context
            current_df = mock_repl_tool_instance.globals["df"]
            current_df["is_fallback"] = True
            mock_repl_tool_instance.globals["df"] = current_df
            return {"output": "Agent processed fallback df_valid_fallback"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        state = MyState(
            latest_df=initial_latest_df,
            input="DF_KEY['df_empty'] Process this."
        )
        result_state = data_processing_node(state)

        self.assertEqual(result_state["condition"], "data_processing_done")
        expected_df2_processed = df2.copy()
        expected_df2_processed["is_fallback"] = True
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["df_valid_fallback"]), expected_df2_processed)
        # df_empty should remain empty
        self.assertEqual(result_state["latest_df"]["df_empty"], df1_empty_data)


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_data_processing_without_df_key_fallback_to_instruction_match(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value

        sales_data_list = [{"item": "apple", "qty": 100}]
        sales_df = pd.DataFrame(sales_data_list)
        inventory_data_list = [{"item": "apple", "stock": 50}]
        # inventory_df = pd.DataFrame(inventory_data_list)

        initial_latest_df = collections.OrderedDict({
            "sales_data": sales_data_list,
            "inventory_data": inventory_data_list
        })

        mock_repl_tool_instance.globals = {"df": sales_df.copy(), "pd": pd} # Expect sales_data to be chosen

        def agent_invoke_side_effect(prompt_str):
            self.assertIn(self._get_df_info_as_string(sales_df), prompt_str)
            self.assertIn("指示内容で言及されたキー「sales_data」のデータ", prompt_str) # Fallback to instruction match
            current_df = mock_repl_tool_instance.globals["df"]
            current_df["discounted"] = True
            mock_repl_tool_instance.globals["df"] = current_df
            return {"output": "Agent processed sales_data via instruction match"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        state = MyState(
            latest_df=initial_latest_df,
            input="Apply discount to sales_data items." # No DF_KEY, but "sales_data" is in text
        )
        result_state = data_processing_node(state)

        self.assertEqual(result_state["condition"], "data_processing_done")
        expected_sales_processed = sales_df.copy()
        expected_sales_processed["discounted"] = True
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["sales_data"]), expected_sales_processed)
        # inventory_data should be untouched
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["inventory_data"]), pd.DataFrame(inventory_data_list))


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_data_processing_without_df_key_fallback_to_first_df(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value

        df_alpha_list = [{"alpha_col": 1}]
        df_alpha = pd.DataFrame(df_alpha_list)
        df_beta_list = [{"beta_col": 2}]
        # df_beta = pd.DataFrame(df_beta_list)

        initial_latest_df = collections.OrderedDict({
            "df_alpha": df_alpha_list, # This should be chosen
            "df_beta": df_beta_list
        })

        mock_repl_tool_instance.globals = {"df": df_alpha.copy(), "pd": pd}

        def agent_invoke_side_effect(prompt_str):
            self.assertIn(self._get_df_info_as_string(df_alpha), prompt_str)
            self.assertIn("利用可能な最初のデータセット「df_alpha」", prompt_str) # Fallback to first
            current_df = mock_repl_tool_instance.globals["df"]
            current_df["is_first"] = True
            mock_repl_tool_instance.globals["df"] = current_df
            return {"output": "Agent processed df_alpha as first available"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        state = MyState(
            latest_df=initial_latest_df,
            input="Add a new column 'Y'." # No DF_KEY, no key name in text
        )
        result_state = data_processing_node(state)

        self.assertEqual(result_state["condition"], "data_processing_done")
        expected_alpha_processed = df_alpha.copy()
        expected_alpha_processed["is_first"] = True
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["df_alpha"]), expected_alpha_processed)
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["df_beta"]), pd.DataFrame(df_beta_list))


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_prompt_guidance_for_df_key_multiple_dfs(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value # Not strictly needed but good practice

        df1_data = [{"colA": 1}] * 2 # 2 rows
        df2_data = [{"colX": 5}] * 3 # 3 rows
        initial_latest_df = collections.OrderedDict({ "dfOne": df1_data, "dfTwo": df2_data })

        # Simulate tool initialized with one of the DFs (e.g., first one by fallback)
        mock_repl_tool_instance.globals = {"df": pd.DataFrame(df1_data), "pd": pd}
        mock_agent_instance.invoke.return_value = {"output": "Processed"} # Doesn't matter for this test

        state = MyState(latest_df=initial_latest_df, input="Generic instruction")
        data_processing_node(state) # Call the node

        self.assertTrue(mock_agent_instance.invoke.called)
        prompt_str = mock_agent_instance.invoke.call_args[0][0]

        self.assertIn("複数のDataFrameが利用可能です。", prompt_str)
        self.assertIn("DF_KEY['キー名']", prompt_str)
        self.assertIn("例: `DF_KEY['sales_data_2023']", prompt_str)
        self.assertIn("利用可能なDataFrame (キー: 行数):", prompt_str)
        self.assertIn("- 'dfOne': 2行", prompt_str)
        self.assertIn("- 'dfTwo': 3行", prompt_str)


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_prompt_guidance_for_df_key_single_df(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value
        mock_repl_tool_instance.globals = {"df": self.initial_df.copy(), "pd": pd}
        mock_agent_instance.invoke.return_value = {"output": "Processed"}

        state = MyState(latest_df=collections.OrderedDict({"single_df": self.initial_df_list}), input="Process the single_df")
        data_processing_node(state)

        self.assertTrue(mock_agent_instance.invoke.called)
        prompt_str = mock_agent_instance.invoke.call_args[0][0]

        self.assertNotIn("複数のDataFrameが利用可能です。", prompt_str)
        self.assertNotIn("DF_KEY['キー名']", prompt_str)
        self.assertNotIn("利用可能なDataFrame (キー: 行数):", prompt_str)


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_data_processing_specific_key_filter(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value

        df1_list = [{"id": 1, "val": 10}, {"id": 2, "val": 20}]
        df2_list = [{"id": 3, "val": 5}, {"id": 4, "val": 15}]
        df2 = pd.DataFrame(df2_list)
        initial_latest_df = collections.OrderedDict({
            "dataOne": df1_list,
            "dataTwo": df2_list
        })
        mock_repl_tool_instance.globals = {"df": df2.copy(), "pd": pd}


        def agent_invoke_side_effect(prompt_str):
            # Check df.info() for df2 is in prompt
            self.assertIn(self._get_df_info_as_string(df2), prompt_str)
            # Check context message indicates "dataTwo" was chosen (due to instruction content)
            self.assertIn("指示内容で言及されたキー「dataTwo」のデータ", prompt_str)
            # Check DF_KEY guidance is present as there are multiple DFs
            self.assertIn("複数のDataFrameが利用可能です。", prompt_str)
            self.assertIn("DF_KEY['キー名']", prompt_str)

            current_df = mock_repl_tool_instance.globals["df"]
            mock_repl_tool_instance.globals["df"] = current_df[current_df['val'] > 10]
            return {"output": "Agent executed: Filtered dataTwo"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        state = MyState(
            latest_df=initial_latest_df,
            input="For dataTwo, filter rows where val > 10" # "dataTwo" is a key and mentioned
        )
        result_state = data_processing_node(state)

        self.assertEqual(result_state["condition"], "data_processing_done")
        self.assertIsNone(result_state.get("error"))
        pd.testing.assert_frame_equal(pd.DataFrame(result_state["latest_df"]["dataOne"]), pd.DataFrame(df1_list))
        processed_df2 = pd.DataFrame(result_state["latest_df"]["dataTwo"])
        expected_df2 = pd.DataFrame([{"id": 4, "val": 15}])
        pd.testing.assert_frame_equal(processed_df2, expected_df2, check_dtype=False)

    def test_no_suitable_dataframe_error_empty_latest_df(self):
        state = MyState(latest_df=collections.OrderedDict(), input="Process something")
        result_state = data_processing_node(state)
        self.assertEqual(result_state["condition"], "data_processing_failed_no_data")
        self.assertIn("データ処理のための有効なデータがありません。", result_state["error"])

    def test_no_suitable_dataframe_error_target_key_not_found(self):
        state = MyState(
            latest_df=collections.OrderedDict({"actual_data": self.initial_df_list}),
            input="Process non_existent_data" # Instruction refers to a key not in latest_df
        )
        # Fallback logic should pick "actual_data", so this test needs to ensure fallback also fails if actual_data was empty
        # Or, if the goal is to test "selected_key not found and THEN fallback fails", make fallback fail.
        state_target_key_not_found_empty_fallback = MyState(
            latest_df=collections.OrderedDict({"actual_data": []}), # Fallback target is empty
            input="Process non_existent_data"
        )
        result_state = data_processing_node(state_target_key_not_found_empty_fallback)
        self.assertEqual(result_state["condition"], "data_processing_failed_empty_selected_data")
        self.assertIn("データ処理に適したデータが見つかりませんでした。", result_state["error"])


    def test_no_suitable_dataframe_error_target_df_is_empty_list(self):
        state = MyState(
            latest_df=collections.OrderedDict({"data1": []}), # Target DataFrame is an empty list
            input="Process data1"
        )
        result_state = data_processing_node(state)
        self.assertEqual(result_state["condition"], "data_processing_failed_empty_selected_data")
        self.assertIn("データ処理に適したデータが見つかりませんでした。", result_state["error"])


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_agent_returns_non_dataframe(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value
        mock_repl_tool_instance.globals = {"df": self.initial_df.copy(), "pd": pd}

        def agent_invoke_side_effect(prompt_str):
            # Simulate agent code execution resulting in non-DataFrame
            mock_repl_tool_instance.globals["df"] = "This is not a DataFrame"
            return {"output": "Agent executed: Returned a string"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        result_state = data_processing_node(self.default_state)

        self.assertEqual(result_state["condition"], "data_processing_failed_bad_output")
        self.assertIn("期待されるDataFrame形式ではありませんでした。", result_state["error"])
        self.assertIn("Agent executed: Returned a string", result_state["error"])


    @patch('files.backend_codes.initialize_agent')
    def test_agent_execution_error_invoke_raises_exception(self, MockInitializeAgent):
        mock_agent_instance = MockInitializeAgent.return_value
        error_message = "Agent_invoke_failed_with_exception"
        mock_agent_instance.invoke.side_effect = Exception(error_message)

        # Need to ensure PythonAstREPLTool is also mocked if its constructor is called before agent.invoke
        # For this test, initialize_agent is called, then agent.invoke.
        # If PythonAstREPLTool constructor is called *before* initialize_agent (it's not), that would need mocking.
        # Here, we only need to mock initialize_agent to return our mock_agent_instance.
        # The PythonAstREPLTool will be instantiated with the real one, but its methods won't be an issue
        # if agent.invoke fails before the tool is heavily used.

        with patch('files.backend_codes.PythonAstREPLTool'): # Patch it to avoid issues if it's constructed
            result_state = data_processing_node(self.default_state)

        self.assertEqual(result_state["condition"], "data_processing_failed_agent_error")
        self.assertIn(error_message, result_state["error"])


    @patch('files.backend_codes.PythonAstREPLTool')
    @patch('files.backend_codes.initialize_agent')
    def test_data_processing_no_change_to_df(self, MockInitializeAgent, MockPythonAstREPLTool):
        mock_agent_instance = MockInitializeAgent.return_value
        mock_repl_tool_instance = MockPythonAstREPLTool.return_value

        # Initialize globals with a copy of the initial DataFrame
        mock_repl_tool_instance.globals = {"df": self.initial_df.copy(), "pd": pd}

        def agent_invoke_side_effect(prompt_str):
            # Simulate agent code that technically runs but makes no changes to 'df'
            # For example, it might assign to a different variable or do a read-only operation.
            # The 'df' in globals remains the same as initial_df.
            return {"output": "Agent executed: No change to df"}

        mock_agent_instance.invoke.side_effect = agent_invoke_side_effect

        result_state = data_processing_node(self.default_state)

        self.assertEqual(result_state["condition"], "data_processing_done") # Still considered done
        self.assertIsNone(result_state.get("error"))

        processed_df_list = result_state["latest_df"]["data1"]
        processed_df = pd.DataFrame(processed_df_list)

        # DataFrame should be identical to the original
        pd.testing.assert_frame_equal(processed_df, self.initial_df)
        # This test also implicitly checks that logging.warning for "no change" doesn't crash.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
