
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path: Path):
    # Load the source document.
    loader = TextLoader(
        str(file_path),
        encoding="utf-8",
    )
    documents = loader.load()

    # Split the loaded document into chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=70,
        chunk_overlap=15,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# Locate the company policy document.
document_path = (
    Path(__file__).parent
    / "documents"
    / "company_policy.txt"
)

# Run the preprocessing pipeline.
chunks = load_and_split_document(document_path)

print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk.page_content)
    print("Source:", chunk.metadata["source"])
    print("Start index:", chunk.metadata["start_index"])
