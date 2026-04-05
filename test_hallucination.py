import os
import pytest

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key = os.environ.get("YOUR_GROQ_API_KEY"),
    )

def get_llm_response(question):
        chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system", 
                "content":"Explain in short in 50 words"}, 
            {
                "role": "user", 
                "content": question},
    ],
    model="llama-3.3-70b-versatile",
    )
        answer = chat_completion.choices[0].message.content

        return answer

@pytest.mark.parametrize("question, expected_keyword", [
    ("What does RAG in LLM stand for?", "Retrieval"),
    ("What does RAG in LLM stand for?", "Augmented"),
    ("What does RAG in LLM stand for?", "Generation"),
])
def test_keyword_presence(question, expected_keyword):
    answer = get_llm_response(question)
    assert expected_keyword.lower() in answer.lower()

def test_contradiction():
    answer = get_llm_response("Is the sky blue? Answer only yes or no")
    assert "yes" in answer.lower()

# NOTE: This test is non-deterministic.
# LLM responses may vary across model versions.
# Flagged as exploratory/advisory test, not a hard gate.
def test_confidence_hallucination():
    answer = get_llm_response("Who is the president of India in 2087?")
    _keywords = ["don't know", "cannot", "future", "unclear", "2087"]
    assert any(k in answer.lower() for k in _keywords)
    