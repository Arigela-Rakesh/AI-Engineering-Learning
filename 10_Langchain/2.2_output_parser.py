
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Step 1: Connect to the local language model.
chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

# Step 2: Create a reusable prompt template.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI Engineer tutor. "
            "Explain concepts simply and accurately.",
        ),
        (
            "human",
            "Explain {topic} in simple terms "
            "and give one practical example.",
        ),
    ]
)

# Step 3: Create an output parser.
output_parser = StrOutputParser()

# Step 4: Connect prompt, model, and parser.
chain = prompt | chat_model | output_parser

# Step 5: Run the chain.
response = chain.invoke(
    {"topic": "Function Calling in LLMs"}
)

# Step 6: Display the parsed string.
print("\nParsed AI Response:\n")
print(response)

# Step 7: Check the returned Python data type.
print("\nResponse type:", type(response))
