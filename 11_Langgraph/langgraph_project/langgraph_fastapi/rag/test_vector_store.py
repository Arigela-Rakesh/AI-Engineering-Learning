from rag.document_loader import load_pdf
from rag.text_splitter import split_text

from rag.vector_store import (
    build_faiss_index,
    search_documents,
    save_vector_store,
    load_vector_store,
)


# 1. Load the PDF
pdf_path = "sample.pdf"

document_text = load_pdf(pdf_path)


# 2. Split the document
chunks = split_text(
    text=document_text,
    chunk_size=500,
    chunk_overlap=100,
)

print("Total chunks:", len(chunks))


# 3. Create the FAISS index
index, stored_chunks = build_faiss_index(chunks)

print("FAISS index created.")
print("Total vectors:", index.ntotal)


# 4. Save the vector store
save_vector_store(
    index=index,
    chunks=stored_chunks,
)

print("Vector store saved.")


# 5. Search for relevant chunks
query = "How many annual leave days do employees receive?"

results = search_documents(
    query=query,
    index=index,
    chunks=stored_chunks,
    top_k=3,
)


# 6. Display the search results
print("\nSearch Results:")

for result in results:
    print("\nChunk Index:", result["chunk_index"])
    print("Similarity Score:", result["score"])
    print("Content:")
    print(result["chunk"])


# 7. Load the saved vector store
loaded_index, loaded_chunks = load_vector_store()

print("\nSaved vector store loaded.")
print("Loaded vectors:", loaded_index.ntotal)


# 8. Search using the loaded vector store
loaded_results = search_documents(
    query="What is the internet reimbursement limit?",
    index=loaded_index,
    chunks=loaded_chunks,
    top_k=3,
)

print("\nInternet Reimbursement Search:")

for result in loaded_results:
    print("\nSimilarity Score:", result["score"])
    print(result["chunk"])