import pytest
import deepeval
from deepeval import assert_test
#from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.dataset import EvaluationDataset
import os
import json
import openai
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ['OPENAI_API_KEY']
#client = OpenAI()

# Collect all problems and answers
inputs = []
outputs = []
expected_outputs = []
for i in range(12,13):
    path = './gpt4_problems/set'+str(i)+'/'
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)
            with open (file_path, 'r') as json_file:
                data = json.load(json_file)
                messages = data['messages']
                if len(messages) >= 4:
                    inputs.append(messages[1]['content'])
                    expected_outputs.append(messages[3]['content'])
# print("Inputs are: ")
# print(inputs)
# print("Expected outputs are: ")
# print(expected_outputs[0])
with open('inputs_copy.txt', 'w') as file:
    file.write(str(inputs))
with open('expected_outputs_copy.txt', 'w') as file:
    file.write(str(expected_outputs))


# To run this file: deepeval test run <file_name>.

# dataset = EvaluationDataset(alias="My dataset", test_cases=[])


# @pytest.mark.parametrize(
#     "test_case",
#     dataset,
# )
    


i=0
while(i < len(inputs)):
    response = openai.chat.completions.create(model='gpt-3.5-turbo',
                                              messages = [{"role": "user", "content": inputs[i]}, {"role": "user", "content": "Answer the questions and explain your reasoning in the format of\n \"Answer1: x \n Answer2: x \n Explanation1: x \n Explanation2: x\n \""}],
                                              temperature = 0)
    #import pdb; pdb.set_trace()
    output = str(response.choices[0].message.content).strip()
    if output != "" and output != None:
        outputs.append(output)
    i += 1


# for i in range (len(outputs)):
#     print(str(i+1)+": "+outputs[i])
testcases = []

for i in range(len(inputs)):
    test_case = LLMTestCase(
        input=inputs[i],
        # Replace this with the actual output of your LLM application
        actual_output=outputs[i],
        expected_output=expected_outputs[i]
    )
    testcases.append(test_case)


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


