import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric , GEval

# import judge from test_deepeval.py
from test_deepeval1 import judge

# import groq client
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key = os.environ.get("YOUR_GROQ_API_KEY"),
    )

def get_llm_response(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# dataset with multiple test cases
dataset = EvaluationDataset([   
    Golden(
        input="What are symptoms of high blood pressure?",
        #actual_output="Symptoms include headaches and dizziness.",
        expected_output="Symptoms include headaches, dizziness, and shortness of breath.",
        retrieval_context=["High blood pressure symptoms include headaches, dizziness, and shortness of breath."]
    ),
    Golden(
        input="What causes diabetes?",
        #actual_output="Diabetes is caused by insufficient insulin production.",
        expected_output="Diabetes is caused by insufficient insulin production or insulin resistance.",
        retrieval_context=["Diabetes occurs when the pancreas doesn't produce enough insulin or the body can't use insulin effectively."]
    )
])

# metrics
metrics = [
    FaithfulnessMetric(threshold = 0.7, model=judge),
    AnswerRelevancyMetric(threshold=0.7, model=judge),
    ContextualRecallMetric(threshold=0.7, model=judge),
    GEval(
        name = "Completeness",
        criteria = "Check if the actual output covers all important points from the retrieval context.",
        evaluation_params = [
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
      ],
        threshold = 0.7,
        model = judge
     )
]

#Pytest function
@pytest.mark.parametrize("golden", dataset.goldens)
def test_med_chatbot(golden: Golden):
    answer = get_llm_response(golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output = answer,
        expected_output=golden.expected_output,
        retrieval_context=golden.retrieval_context
    )
    assert_test(test_case, metrics)
