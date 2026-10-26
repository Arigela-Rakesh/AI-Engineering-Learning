
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Step 1: Locate and load the document.
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


# Step 2: Create the text splitter.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=15,
    length_function=len,
    add_start_index=True,
)

# Step 3: Split the loaded documents.
chunks = text_splitter.split_documents(documents)


# Step 4: Display the results.
print("Original document count:", len(documents))
print("Number of chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print("Content:")
    print(chunk.page_content)
    print("Character count:", len(chunk.page_content))
    print("Metadata:", chunk.metadata)
