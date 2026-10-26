
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama


# Step 1: Define the expected output structure.
class CustomerAnalysis(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the customer message"
    )
    issue: str = Field(
        description="Main customer issue, or 'no issue' if none exists"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="Urgency level of the customer message"
    )


# Step 2: Create the parser.
output_parser = PydanticOutputParser(
    pydantic_object=CustomerAnalysis
)


# Step 3: Connect to Ollama.
chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# Step 4: Create a prompt with an explicit JSON example.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze customer support messages. "
            "Return one JSON object containing actual analysis values. "
            "Do not return the schema, property descriptions, or "
            "formatting instructions. Do not use Markdown fences. "
            "Use only the allowed sentiment and urgency values. "
            "If no issue exists, set issue to 'no issue'."
        ),
        (
            "human",
            """Analyze this customer message:

{customer_message}

Return actual values in this JSON structure:
{{
  "sentiment": "negative",
  "issue": "late delivery",
  "urgency": "medium"
}}

The example values are illustrative. Choose values that match
the customer's message.

{format_instructions}"""
        ),
    ]
)


# Step 5: Supply parser instructions.
prompt = prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)


# Step 6: Connect prompt, model, and parser.
chain = prompt | chat_model | output_parser


# Step 7: Analyze the customer message.
try:
    result = chain.invoke(
        {
            "customer_message": (
                "My payment was deducted twice, "
                "and I need help immediately."
            )
        }
    )

    # Step 8: Display the validated result.
    print("\nValidated Customer Analysis:\n")
    print(result)

    print("\nSentiment:", result.sentiment)
    print("Issue:", result.issue)
    print("Urgency:", result.urgency)
    print("Response type:", type(result))

except Exception as error:
    print("\nCould not parse the model response.")
    print("Error:", error)
