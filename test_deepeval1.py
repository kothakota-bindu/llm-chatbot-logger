from groq import Groq
from dotenv import load_dotenv
import os
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, GEval
from deepeval.models import DeepEvalBaseLLM 

class GroqJudge(DeepEvalBaseLLM):
     def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.environ.get("YOUR_GROQ_API_KEY"),
                           )

     def get_model_name(self):
        return "llama-3.3-70b-versatile"
     
     def load_model(self):
        return self.client
     
     def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

     async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
     pass

judge = GroqJudge()

# deepeval Test
test_case = LLMTestCase(
    input = "What are symptoms of high blood pressure?",
    actual_output = "Symptoms include headaches and dizziness.",
    expected_output="Symptoms of high blood pressure include headaches, dizziness, and shortness of breath.",
    retrieval_context = ["High blood pressure symptoms include headaches, dizziness, and shortness of breath."]
)

if __name__ == "__main__":
   metric1 = FaithfulnessMetric(threshold = 0.7, model=judge)
   metric1.measure(test_case)

   metric2 = AnswerRelevancyMetric(threshold=0.7, model=judge)
   metric2.measure(test_case)

   metric3 = ContextualRecallMetric(threshold=0.7, model=judge)
   metric3.measure(test_case)

   completeness_metric = GEval(
      name = "Completeness",
      criteria = "Check if the actual output covers all important points from the retrieval context.",
      evaluation_params = [
         LLMTestCaseParams.ACTUAL_OUTPUT,
         LLMTestCaseParams.RETRIEVAL_CONTEXT
      ],
      threshold = 0.7,
      model = judge
)
   completeness_metric.measure(test_case)

   print("FaithfulnessMetric:", metric1.score, " ", metric1.reason)
   print("AnswerRelevancyMetric:", metric2.score, " ", metric2.reason)
   print("ContextualRecallMetric:", metric3.score, " ", metric3.reason)
   print("completeness_metric:", completeness_metric.score, " ", completeness_metric.reason)
