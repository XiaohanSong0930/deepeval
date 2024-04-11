import google.generativeai as genai

genai.configure(api_key="")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# convo = model.start_chat(history=[
# ])

# convo.send_message("Hello")
# print(convo.last.text)

import pytest
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.dataset import EvaluationDataset
import os
import json
import openai
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ['OPENAI_API_KEY']

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

with open('inputs_copy.txt', 'w') as file:
    file.write(str(inputs))
with open('expected_outputs_copy.txt', 'w') as file:
    file.write(str(expected_outputs))


i=0
while(i < len(inputs)):
    convo = model.start_chat(history=[])
    convo.send_message(inputs[i])
    convo.send_message("Answer the questions and explain your reasoning in the format of\n \"Answer1: x \n Answer2: x \n Explanation1: x \n Explanation2: x\n \"")
    output = convo.last.text
    if output != "" and output != None:
        outputs.append(output)
    i += 1


testcases = []

for i in range(len(inputs)):
    test_case = LLMTestCase(
        input=inputs[i],
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


