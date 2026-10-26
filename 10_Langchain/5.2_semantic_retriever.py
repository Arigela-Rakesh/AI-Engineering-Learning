
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


# 2. Split the document into chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=15,
    add_start_index=True,
)

chunks = splitter.split_documents(documents)


# 3. Load the embedding model.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create a FAISS vector store.
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model,
)


# 5. Convert the vector store into a retriever.
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# 6. Ask a question.
question = "How much vacation time can employees take each year?"

results = retriever.invoke(question)


# 7. Display retrieved documents.
print("Question:", question)
print("\nRetrieved documents:")

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content)
    print("Source:", document.metadata["source"])
