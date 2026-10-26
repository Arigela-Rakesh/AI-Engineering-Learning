
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Define the saved vector store location.
base_directory = Path(__file__).parent

vector_store_path = (
    base_directory
    / "faiss_company_policy"
)


# 2. Load the embedding model.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 3. Load the saved FAISS vector store.
vector_store = FAISS.load_local(
    str(vector_store_path),
    embedding_model,
    allow_dangerous_deserialization=True,
)


# 4. Convert the vector store into a retriever.
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# 5. Ask a question.
question = "How much vacation time can employees take each year?"

results = retriever.invoke(question)


# 6. Display the retrieved documents.
print("Question:", question)
print("\nRetrieved documents:")

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content)
    print("Source:", document.metadata["source"])

print("\nLoaded and searched the saved vector store successfully!")
