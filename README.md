# LLM Chatbot with Response Logger

A simple command-line chatbot built with Groq API (LLaMA 3.3 70B) that logs every conversation to a CSV file with response time tracking.

## Setup

1. Clone the repo
2. Install dependencies
   pip install groq python-dotenv
3. Create a `.env` file in the root folder
   GROQ_API_KEY=your_actual_key_here
4. Run the script
   python chatbot.py

## Usage

- Type any question and press Enter
- Type `exit` to quit
- Logs are saved automatically to `chat_logs.csv`

## CSV Output

<img width="1294" height="907" alt="image" src="https://github.com/user-attachments/assets/5274fc09-c7f7-4505-8401-3b40de4b3eb6" />


## Tech Stack

- Python 3.10+
- Groq API (LLaMA 3.3 70B)
- python-dotenv
