
from pathlib import Path

from langchain_community.document_loaders import TextLoader


# Step 1: Define the document path.
file_path = (
    Path(__file__).parent
    / "documents"
    / "company_policy.txt"
)

# Step 2: Create the document loader.
loader = TextLoader(
    str(file_path),
    encoding="utf-8",
)

# Step 3: Load the document.
documents = loader.load()

# Step 4: Display the loaded documents.
print("Number of documents:", len(documents))

for document in documents:
    print("\nDocument content:\n")
    print(document.page_content)

    print("\nDocument metadata:")
    print(document.metadata)
