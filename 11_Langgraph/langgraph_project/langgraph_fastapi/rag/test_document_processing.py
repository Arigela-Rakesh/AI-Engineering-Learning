from rag.document_loader import load_pdf
from rag.text_splitter import split_text


pdf_path = "sample.pdf"

# 1. Extract text from PDF
document_text = load_pdf(pdf_path)

print("\nExtracted Text:")
print(document_text[:1000])

# 2. Split extracted text
chunks = split_text(
    text=document_text,
    chunk_size=500,
    chunk_overlap=100,
)

print("\nTotal Chunks:", len(chunks))

# 3. Display chunks
for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)