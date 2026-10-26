
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the company policy document.
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


# 2. Split the document.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=15,
    add_start_index=True,
)

chunks = splitter.split_documents(documents)


# 3. Create embeddings.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create the FAISS vector store.
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model,
)


# 5. Search and return distance scores.
question = "What is the company's policy for office parking?"

results = vector_store.similarity_search_with_score(
    question,
    k=3,
)


# 6. Display the results and scores.
print("Question:", question)
print("\nRetrieved documents with scores:")

for index, result in enumerate(results, start=1):
    document = result[0]
    score = result[1]

    print(f"\n--- Result {index} ---")
    print("Content:", document.page_content)
    print("Distance score:", score)
