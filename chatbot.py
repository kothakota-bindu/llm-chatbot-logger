import os
import csv
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key = os.environ.get("YOUR_GROQ_API_KEY"),
    )

with open("chat_logs.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Answer", "Response Time (s)"])
   
while(True):
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    start = time.time()

    chat_completion = client.chat.completions.create(
    messages = [
        {
            "role": "system", 
            "content":"Explain in short in 50 words"},
        {
            "role": "user", 
            "content": user_input},
    ],
    model="llama-3.3-70b-versatile",
)

    end = time.time()
    response_time = round(end - start, 2)

    answer = chat_completion.choices[0].message.content
    print(answer)
    print(response_time)

    with open("chat_logs.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_input, answer, response_time])