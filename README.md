---
title: Chat with MySQL
emoji: 💬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# Chat with MySQL

This is a Streamlit application that allows users to interact with a MySQL database via natural language queries. The app uses LangChain, Groq, and Streamlit to generate SQL queries and respond with database results in natural language.

## Features
- Connect to your MySQL database and chat with it using natural language.
- Automatically generate SQL queries based on your questions.
- Receive responses both in SQL and human-readable formats.

## Libraries and Tools Used
- **dotenv**: Loads environment variables from a `.env` file.
- **LangChain**: Handles the prompt templates and chains for generating SQL queries and responses.
- **Groq**: Utilized as the model for chat-based interactions and SQL generation.
- **Streamlit**: Provides the interface for interacting with the database and handling the conversation.
- **SQLDatabase**: LangChain's utility to manage SQL database connections and queries.
  
## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/chat-with-mysql.git
   cd chat-with-mysql
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Create a .env file in the root directory of the project and add your database credentials:

bash
Copy code
DB_USER=root
DB_PASSWORD=admin
DB_HOST=localhost
DB_PORT=3306
DB_NAME=Chinook
Run the application:

bash
Copy code
streamlit run app.py
Open your browser and go to the Streamlit web app, typically at http://localhost:8501.

How It Works
The app connects to a MySQL database using credentials from environment variables.
It uses a LangChain model to process user queries, convert them into SQL statements, and return the results.
You can view the SQL query generated from your questions and the corresponding response.
Configuration
Check out the configuration reference at Hugging Face Spaces Config Reference.

