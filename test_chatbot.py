import os
import time
import pytest

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key = os.environ.get("YOUR_GROQ_API_KEY"),
    )

def get_llm_response(question):
        start = time.time()

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

        end = time.time()
        response_time = round(end - start, 2)
        answer = chat_completion.choices[0].message.content

        return answer, response_time

@pytest.fixture(scope="module")
def llm_response():
    return get_llm_response("What is RAG?")

def test_empty_response(llm_response):
    answer, _ = llm_response
    assert answer is not None
    assert answer != "" 

def test_response_time(llm_response):
    _, response_time = llm_response
    assert response_time < 10

def test_answer_words_count(llm_response):
    answer, _ = llm_response
    assert len(answer.split()) < 100

def test_answer_type(llm_response):
    answer, _ = llm_response
    assert isinstance(answer, str)
