
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Define paths.
base_directory = Path(__file__).parent

document_path = (
    base_directory
    / "documents"
    / "company_policy.txt"
)

vector_store_path = (
    base_directory
    / "faiss_company_policy"
)


# 2. Load the document.
loader = TextLoader(
    str(document_path),
    encoding="utf-8",
)

documents = loader.load()


# 3. Split the document.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=15,
    add_start_index=True,
)

chunks = splitter.split_documents(documents)


# 4. Load the embedding model.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 5. Create the FAISS vector store.
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model,
)


# 6. Save the vector store.
vector_store.save_local(str(vector_store_path))

print("Vector store saved successfully!")
print("Saved location:", vector_store_path)
