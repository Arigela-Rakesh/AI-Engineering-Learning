
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# Step 1: Create the prompt.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI Engineer tutor. "
            "Explain concepts simply and accurately.",
        ),
        (
            "human",
            "Explain {topic} in simple terms. "
            "Give one practical example.",
        ),
    ]
)


# Step 2: Connect to the local model.
chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Step 3: Create the output parser.
output_parser = StrOutputParser()


# Step 4: Build the LCEL chain.
chain = prompt | chat_model | output_parser


# Step 5: Invoke the chain.
result = chain.invoke(
    {"topic": "LCEL in LangChain"}
)

print("\nFirst response:\n")
print(result)


# Step 6: Reuse the same chain with a different input.
second_result = chain.invoke(
    {"topic": "Pydantic Output Parsers"}
)

third_result=chain.invoke(
    {"topic":"AI Agents"}
)

print("\nSecond response:\n")
print(second_result)


print("\nThird response:\n")
print(third_result)
