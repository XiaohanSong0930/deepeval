from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.dataset import EvaluationDataset
import pytest 

test_case1 = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output of your LLM application
    actual_output="We offer a 30-day full refund.",
    expected_output="hhh"
)

test_case2 = LLMTestCase(
    input="What if these shoes fit?",
    # Replace this with the actual output of your LLM application
    actual_output="Then you're all set",
    expected_output="Then you are good."
)
testcases = []
testcases.append(test_case1)
testcases.append(test_case2)

dataset = EvaluationDataset(
    alias = 'My first dataset',
    test_cases= testcases
)

@pytest.mark.parametrize(
        'test_case',
        dataset,
)

def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    correctness_metric = GEval(
        name='Correctness',
        criteria="Correctness - determine if the actual output is correct according to the expected output.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        strict_mode=True,
    )
    assert_test(test_case, [answer_relevancy_metric, correctness_metric])

# def test_answer_relevancy():
#     answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.6)
#     test_case1 = LLMTestCase(
#         input="What if these shoes don't fit?",
#         # Replace this with the actual output of your LLM application
#         actual_output="We offer a 30-day full refund.",
#         expected_output="hhh"
#     )
#     test_case2 = LLMTestCase(
#         input="What if these shoes fit?",
#         # Replace this with the actual output of your LLM application
#         actual_output="Then you're all set",
#         expected_output="Then you are good."
#     )
#     correctness_metric = GEval(
#         name="Correctness",
#         criteria="Correctness - determine if the actual output is correct according to the expected output.",
#         evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
#         strict_mode=True,
#     )
#     assert_test(test_case, [answer_relevancy_metric, correctness_metric])