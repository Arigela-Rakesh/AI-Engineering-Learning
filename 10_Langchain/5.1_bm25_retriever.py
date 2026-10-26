
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Locate and load the document.
file_path = (
    Path(__file__).parent
    / "documents"
    / "company_policy.txt"
)

loader = TextLoader(
    str(file_path),
    encoding="utf-8",
)

documents = loader.load()


# 2. Split the document into chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=15,
    add_start_index=True,
)

chunks = splitter.split_documents(documents)


# 3. Create a BM25 retriever.
retriever = BM25Retriever.from_documents(chunks)

# Return the three most relevant chunks.
retriever.k = 3


# 4. Ask a question.
question = "What security rules apply to remote work?"

results = retriever.invoke(question)


# 5. Display the retrieved chunks.
print("Question:", question)
print("\nRetrieved documents:")

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content)
    print("Source:", document.metadata["source"])
