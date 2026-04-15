import json
from deepeval.dataset import Golden, EvaluationDataset

with open ("test_cases.json") as f:
    data = json.load(f)

goldens = [
    Golden(
        input=item["input"],
        expected_output=item["expected_output"],
        retrieval_context=item["retrieval_context"]
    )
    for item in data
]

dataset = EvaluationDataset(goldens)
