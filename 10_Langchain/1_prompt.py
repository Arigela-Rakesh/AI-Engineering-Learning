from langchain_core.prompts import PromptTemplate
prompt=PromptTemplate.from_template(
    "You are an AI Engineer tutor. Explain {topic} "
    "in simple terms and give one practical example."
)
formatted_prompt=prompt.format(
    topic="Retrieval-Augmented Generation (RAG)"
)

print(formatted_prompt)