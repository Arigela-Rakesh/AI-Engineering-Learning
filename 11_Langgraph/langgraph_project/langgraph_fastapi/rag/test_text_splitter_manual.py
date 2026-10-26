from rag.text_splitter import split_text


sample_text = """
Employee Leave Policy

Employees receive 20 annual leave days per year.
Leave requests must be approved by the manager.

Internet Reimbursement Policy

Employees can claim up to Rs. 1500 per month.
Claims require valid supporting documents.

FAQ

How many annual leave days are provided?
Employees receive 20 annual leave days per year.

What is the internet reimbursement amount?
Employees can claim up to Rs. 1500 per month.
"""

chunks = split_text(
    sample_text,
    chunk_size=200,
    chunk_overlap=30,
)

for index, chunk in enumerate(chunks):
    print(f"\n--- Chunk {index} ---")
    print(chunk)