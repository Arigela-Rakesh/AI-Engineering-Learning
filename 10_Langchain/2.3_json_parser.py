
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

# Step 1: Connect to the local model.
chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    format="json",
)

# Step 2: Create the JSON output parser.
output_parser = JsonOutputParser()

# Step 3: Create a prompt that specifies the JSON structure.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a customer support message analyzer. "
            "Return only valid JSON. Do not include Markdown "
            "or any text outside the JSON object.",
        ),
        (
            "human",
            """Analyze this customer message:
{customer_message}

Return a JSON object with these fields:
- sentiment: positive, neutral, or negative
- issue: a short description of the main issue
- urgency: low, medium, or high

{format_instructions}""",
        ),
    ]
)

# Step 4: Supply the parser's format instructions.
prompt = prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)

# Step 5: Connect the prompt, model, and parser.
chain = prompt | chat_model | output_parser

# Step 6: Invoke the chain with a customer message.
result = chain.invoke(
    {
        "customer_message": (
            "The product is excellent, and it arrived earlier than expected!"
        )
    }
)

# Step 7: Display the parsed result.
print("\nParsed JSON Result:\n")
print(result)

print("\nResponse type:", type(result))
print("Sentiment:", result["sentiment"])
print("Issue:", result["issue"])
print("Urgency:", result["urgency"])
