# Import necessary libraries and modules for various tasks
from dotenv import load_dotenv  # For loading environment variables from a .env file (e.g., database credentials)
from langchain_core.messages import AIMessage, HumanMessage  # For handling AI and user messages
from langchain_core.prompts import ChatPromptTemplate  # To create templates for chatbot responses
from langchain_core.runnables import RunnablePassthrough  # To chain different operations (e.g., inputs/outputs)
from langchain_community.utilities import SQLDatabase  # Utility to connect to SQL databases using LangChain
from langchain_core.output_parsers import StrOutputParser  # To parse outputs into plain text
from langchain_groq import ChatGroq  # Integrates the Groq model for generating chat responses
import streamlit as st  # Streamlit for building the web interface
import os  # For accessing environment variables (e.g., credentials or other settings)
import psycopg2  # PostgreSQL database adapter for database connections

# Load environment variables (e.g., database credentials) from the .env file
load_dotenv()

# Function to initialize the database connection
def init_database() -> SQLDatabase:
    try:
        # Retrieve connection details from environment variables or set defaults
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "beekhani143")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "")

        # Construct the database URI for PostgreSQL connection
        db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        
        # Connect to the database using SQLDatabase utility and return the connection
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        # If connection fails, display error in the Streamlit UI
        st.error(f"Failed to connect to the database: {e}")
        return None

# Function to generate the SQL query chain based on user input and conversation history
def get_sql_chain(db):
    # Define the prompt template for generating SQL queries based on schema and chat history
    template = """
    You are a data analyst. You are interacting with a user who is asking questions about the company's database.
    Based on the table schema below, write a SQL query that answers the user's question. Consider the conversation history.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    Write only the SQL query and nothing else.
    
    Question: {question}
    SQL Query:
    """
    
    # Create the prompt template from the instructions above
    prompt = ChatPromptTemplate.from_template(template)
    # Initialize the Groq model for generating SQL responses (deterministic output with temperature=0)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    # Function to retrieve the database schema (table structure)
    def get_schema(_):
        return db.get_table_info()
    
    # Build a chain of operations: get the schema, generate SQL query, and parse as plain text
    return (
        RunnablePassthrough.assign(schema=get_schema)  # Pass schema to the chain
        | prompt  # Use the prompt template to guide query creation
        | llm  # Generate SQL query using the Groq model
        | StrOutputParser()  # Parse the output as a plain text SQL query
    )

# Function to generate a natural language response based on SQL query and database results
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    # Generate the SQL query using the SQL chain
    sql_chain = get_sql_chain(db)

    # Define the template for generating natural language responses
    template = """
    You are a data analyst. Based on the schema, SQL query, and response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User Question: {question}
    SQL Response: {response}
    """
    
    # Create the prompt template for generating a natural language response
    prompt = ChatPromptTemplate.from_template(template)
    # Initialize the Groq model for response generation
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    # Chain the steps: Generate SQL, execute it, and generate the natural language response
    chain = (
        RunnablePassthrough.assign(query=sql_chain)  # Generate SQL query
        .assign(
            schema=lambda _: db.get_table_info(),  # Pass the schema for query execution
            response=lambda vars: db.run(vars["query"].replace("\\", "")),  # Execute the query and clean the output
        )
        | prompt  # Generate a response from the AI using the prompt
        | llm  # Get the final response from the model
        | StrOutputParser()  # Parse the output into plain text
    )

    # Invoke the chain to generate the response
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

    # Return the final natural language response
    return result

# Initialize the chat session in Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        # Initial greeting from the AI assistant
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Configure the Streamlit page title and icon
st.set_page_config(page_title="Chat with PostgreSQL", page_icon=":speech_balloon:")
st.title("Chat with PostgreSQL")  # Display the main title on the web page

# Sidebar for database connection settings
with st.sidebar:
    st.subheader("Settings")  # Display settings section header
    st.write("Connect to your database and start chatting.")  # Instructions for users
    
    # Input fields for database connection details (host, port, user, password, and database name)
    host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"))
    port = st.text_input("Port", value=os.getenv("DB_PORT", "5432"))
    user = st.text_input("User", value=os.getenv("DB_USER", "postgres"))
    password = st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD", "beekhani143"))
    database = st.text_input("Database", value=os.getenv("DB_NAME", "db"))
    
    # Button to initiate database connection
    if st.button("Connect"):
        with st.spinner("Connecting to the database..."):
            db = init_database()  # Attempt to connect to the database
            if db:
                st.session_state.db = db  # Save the connection to session state
                st.success("Connected to the database!")  # Display success message
            else:
                st.error("Connection failed. Please check your settings.")  # Error message if connection fails

# Display the chat history (both AI and user messages)
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Input field for the user to type a message
user_query = st.chat_input("Type a message...")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))  # Save user query

    with st.chat_message("Human"):  # Display user's message in chat
        st.markdown(user_query)
        
    with st.chat_message("AI"):  # Generate and display AI's response
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)  # Get AI response
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))  # Save AI's response to chat history
