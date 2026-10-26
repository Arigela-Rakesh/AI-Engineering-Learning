
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Connect to the model running locally through Ollama.
chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

# Create a reusable chat prompt.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI Engineer tutor. "
            "Explain concepts clearly and accurately.",
        ),
        (
            "human",
            "Explain {topic} in simple terms, give two practical examples, and mention one real-world use case.",
        ),
    ]
)

# Combine the prompt and model into a LangChain chain.
chain = prompt | chat_model

# Invoke the chain with a topic.
response = chain.invoke(
    {"topic": "Vector Databases"}
)

print("\nAI Response:\n")
print(response.content)
