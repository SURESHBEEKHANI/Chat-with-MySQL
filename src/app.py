# Import necessary libraries and modules for various tasks
from dotenv import load_dotenv  # For loading environment variables from a .env file (such as database credentials)
from langchain_core.messages import AIMessage, HumanMessage  # For handling messages from the AI and user
from langchain_core.prompts import ChatPromptTemplate  # To create templates that will guide the chatbot's responses
from langchain_core.runnables import RunnablePassthrough  # To enable chaining of different operations (like inputs/outputs)
from langchain_community.utilities import SQLDatabase  # A tool to help connect to SQL databases using LangChain
from langchain_core.output_parsers import StrOutputParser  # To parse outputs into plain text
from langchain_groq import ChatGroq  # This integrates the Groq model for generating chat responses
import streamlit as st  # Streamlit is used for building the web app (user interface)
import os  # To access environment variables (e.g., credentials or other settings)
import psycopg2  # A PostgreSQL database adapter to enable connections to the database

# Load environment variables (such as DB credentials) from the .env file
load_dotenv()

# Function to establish a connection to the PostgreSQL database
def init_database() -> SQLDatabase:
    try:
        # Retrieve database connection details from environment variables, or set default values
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "beekhani143")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "")

        # Construct the database URI (a URL-like string) with the necessary credentials for PostgreSQL
        db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        
        # Connect to the database using the SQLDatabase utility and return the instance
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        # If connection fails, display an error message on the Streamlit UI
        st.error(f"Failed to connect to database: {e}")
        return None

# Function to create a process (chain) that generates SQL queries based on user input and previous conversation
def get_sql_chain(db):
    # Template to guide how SQL queries are generated. The bot receives table schema and conversation history.
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    Write only the SQL query and nothing else.
    
    Question: {question}
    SQL Query:
    """
    
    # Create a prompt template from the above instructions
    prompt = ChatPromptTemplate.from_template(template)
    # Initialize the Groq model for generating responses with low randomness (temperature=0 for more deterministic outputs)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    # Function to get the schema (structure) of the tables in the database
    def get_schema(_):
        return db.get_table_info()
    
    # Create a chain of operations: 
    # 1. First, get the database schema.
    # 2. Then, use the prompt template to guide query creation.
    # 3. Finally, parse the output as a plain text SQL query.
    return (
        RunnablePassthrough.assign(schema=get_schema)  # Pass the schema into the chain
        | prompt  # Use the prompt template
        | llm  # Generate a response using the Groq model
        | StrOutputParser()  # Parse the response as a string (SQL query)
    )

# Function to generate a natural language response based on SQL query and database result
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    # First, get the SQL chain (responsible for generating SQL queries)
    sql_chain = get_sql_chain(db)

    # Template to guide how the AI responds to the user's query based on the SQL results
    template = """
    You are a data analyst at a company. Based on the table schema, SQL query, and response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    # Create a new prompt template for generating a response
    prompt = ChatPromptTemplate.from_template(template)
    # Initialize the Groq model for response generation
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    # Chain the following tasks:
    # 1. Generate the SQL query using the earlier chain.
    # 2. Get the schema and execute the query on the database.
    # 3. Return the natural language response based on the query and its results.
    chain = (
        RunnablePassthrough.assign(query=sql_chain)  # Generate SQL query
        .assign(
            schema=lambda _: db.get_table_info(),  # Pass the schema to the next step
            response=lambda vars: db.run(vars["query"].replace("\\", "")),  # Execute the SQL query and clean up backslashes
        )
        | prompt  # Use the prompt template for generating natural language response
        | llm  # Generate the final response using the model
        | StrOutputParser()  # Parse the output into plain text
    )

    # Invoke the chain to generate the final response based on the user query and history
    result = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

    # Debugging: Print the SQL query being executed
    if isinstance(result, str):
        print(f"SQL Query: {result}")
    else:
        sql_query = result.get('query', 'No query generated')
        print(f"SQL Query: {sql_query}")

    # Return the result (natural language response)
    return result

# Initialize the chat session when Streamlit app starts
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        # First message from AI assistant
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Streamlit page configuration: Set the page title and icon
st.set_page_config(page_title="Chat with PostgreSQL", page_icon=":speech_balloon:")
st.title("Chat with PostgreSQL")  # Display title on the webpage

# Sidebar configuration for database connection settings
with st.sidebar:
    st.subheader("Settings")  # Display a heading for the settings section
    st.write("Connect to your database and start chatting.")  # Instruction text for users
    
    # Input fields for database connection details (host, port, user, password, and database name)
    host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"))
    port = st.text_input("Port", value=os.getenv("DB_PORT", "5432"))
    user = st.text_input("User", value=os.getenv("DB_USER", "postgres"))
    password = st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD", "beekhani143"))
    database = st.text_input("Database", value=os.getenv("DB_NAME", "db"))
    
    # Button to attempt database connection
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):  # Show a spinner while connecting
            db = init_database()  # Call the function to connect to the database
            if db:
                st.session_state.db = db  # Save the connection in session state
                st.success("Connected to the database!")  # Display success message
            else:
                st.error("Connection failed. Please check your settings.")  # Display error message if connection fails

# Display the chat history (both AI and user messages)
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):  # Display AI messages in the chat
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):  # Display human messages in the chat
            st.markdown(message.content)

# Input field for the user to type their message
user_query = st.chat_input("Type a message...")  # Field to capture user query
if user_query and user_query.strip():  # If the user entered a valid query
    st.session_state.chat_history.append(HumanMessage(content=user_query))  # Save the user query in chat history
    
    with st.chat_message("Human"):  # Display the user's message in the chat
        st.markdown(user_query)
        
    with st.chat_message("AI"):  # Generate and display the AI response
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)  # Get the AI's response
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))  # Add the AI's response to the chat history
