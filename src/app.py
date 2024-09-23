# Import necessary libraries and modules
from dotenv import load_dotenv   # For loading environment variables from .env
from langchain_core.messages import AIMessage, HumanMessage  # Message handling
from langchain_core.prompts import ChatPromptTemplate  # Prompt templates for generating responses
from langchain_core.runnables import RunnablePassthrough  # To chain operations
from langchain_community.utilities import SQLDatabase  # SQL database utility for LangChain
from langchain_core.output_parsers import StrOutputParser  # To parse outputs as strings
# OpenAI model for chat (if used)
from langchain_groq import ChatGroq  # Groq model for chat (currently used)
import streamlit as st  # Streamlit for building the web app
import os  # To access environment variables

# Load environment variables from the .env file (like API keys, database credentials)
load_dotenv()

# Function to initialize a connection to a MySQL database
def init_database() -> SQLDatabase:
    try:
        # Load credentials from environment variables for better security
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "admin")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        database = os.getenv("DB_NAME", "Chinook")

        # Construct the database URI
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        
        # Initialize and return the SQLDatabase instance
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

# Function to create a chain that generates SQL queries from user input and conversation history
def get_sql_chain(db):
    # SQL prompt template
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    Write only the SQL query and nothing else.
    
    Question: {question}
    SQL Query:
    """
    
    # Create a prompt from the above template
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize Groq model for generating SQL queries (can switch to OpenAI if needed)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    # Helper function to get schema info from the database
    def get_schema(_):
        return db.get_table_info()
    
    # Chain of operations: 
    # 1. Assign schema information from the database
    # 2. Use the AI model to generate a SQL query
    # 3. Parse the result into a string
    return (
        RunnablePassthrough.assign(schema=get_schema)  # Get schema info from the database
        | prompt  # Generate SQL query from the prompt template
        | llm  # Use Groq model to process the prompt and return a SQL query
        | StrOutputParser()  # Parse the result as a string
    )

# Function to generate a response in natural language based on the SQL query result
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    # Generate the SQL query using the chain
    sql_chain = get_sql_chain(db)
    
    # Prompt template for natural language response based on SQL query and result
    template = """
    You are a data analyst at a company. Based on the table schema, SQL query, and response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    # Create a natural language response prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize Groq model (alternative: OpenAI)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    # Build a chain: generate SQL query, run it on the database, generate a natural language response
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),  # Get schema info
            response=lambda vars: db.run(vars["query"]),  # Run SQL query on the database
        )
        | prompt  # Use prompt to generate a natural language response
        | llm  # Process prompt with Groq model
        | StrOutputParser()  # Parse the final result as a string
    )
    
    # Execute the chain and return the response
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Initialize the Streamlit session
if "chat_history" not in st.session_state:
    # Initialize chat history with a welcome message from AI
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Set up the Streamlit web page configuration
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

# Streamlit app title
st.title("Chat with MySQL")

# Sidebar for database connection settings
with st.sidebar:
    st.subheader("Settings")
    st.write("Connect to your database and start chatting.")
    
    # Database connection input fields
    host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"))
    port = st.text_input("Port", value=os.getenv("DB_PORT", "3306"))
    user = st.text_input("User", value=os.getenv("DB_USER", "root"))
    password = st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD", "admin"))
    database = st.text_input("Database", value=os.getenv("DB_NAME", "Chinook"))
    
    # Button to connect to the database
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            # Initialize the database connection and store in session state
            db = init_database()
            if db:
                st.session_state.db = db
                st.success("Connected to the database!")
            else:
                st.error("Connection failed. Please check your settings.")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        # Display AI message
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        # Display human message
        with st.chat_message("Human"):
            st.markdown(message.content)

# Input field for user's message
user_query = st.chat_input("Type a message...")
if user_query and user_query.strip():
    # Add user's query to the chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Display user's message in the chat
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    # Generate and display AI's response based on the query
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    # Add AI's response to the chat history
    st.session_state.chat_history.append(AIMessage(content=response))
