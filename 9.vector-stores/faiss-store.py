from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import faiss

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    Document(page_content="Virat Kohli is a batsman for RCB.", metadata={"team": "RCB"}, id="1"),
    Document(page_content="Rohit Sharma is a captain for MI.", metadata={"team": "MI"}, id="2"),
    Document(page_content="MS Dhoni is a captain for CSK.", metadata={"team": "CSK"}, id="3"),
    Document(page_content="Jasprit Bumrah is a bowler for MI.", metadata={"team": "MI"}, id="4"),
    Document(page_content="Ravindra Jadeja is an all-rounder for CSK.", metadata={"team": "CSK"}, id="5"),
]

# Initialize FAISS 
dim = len(embedding.embed_query("test"))  
index = faiss.IndexFlatL2(dim)

vector_store = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add Documents
print("--- Adding Documents ---")
vector_store.add_documents(documents=docs, ids=[d.id for d in docs])

# view all docs
all_docs = vector_store.docstore._dict
print(f"Total documents: {len(all_docs)}")

for doc_id, document in all_docs.items():
    print(f"\nID: {doc_id}")
    print(f"Metadata: {document.metadata}")
    print(f"Content: {document.page_content}")
    
# Search
print("\n--- Similarity Search: Who is a bowler? ---")
results = vector_store.similarity_search("Who among these are a bowler?", k=1)
for res in results:
    print(f"- {res.page_content}")

# Search
print("\n--- Similarity Search: Who is a MI Player? ---")
results = vector_store.similarity_search("Who among these are a MI player?", k=2)
for res in results:
    print(f"- {res.page_content}")

# Metadata Filtering (Show only CSK players)
print("\n--- Filtering for Chennai Super Kings ---")
csk_results = vector_store.similarity_search(
    "Tell me about the players",
    filter={"team": "CSK"},
    k=5
)
for res in csk_results:
    print(f"- {res.page_content}")


# delete Document
print("\n--- Deleting Virat Kohli (ID: 1) ---")
vector_store.delete(ids=["1"])
