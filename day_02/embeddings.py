from sentence_transformers import SentenceTransformer
import numpy as np

class Embeddings :
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.embeddings=None
        self.documents=[]

    def embed(self,documents):
        self.documents.extend(documents)
        self.embeddings = self.model.encode(self.documents,normalize_embeddings=True)

    def search(self,query,k=3):
        query_embedding = self.model.encode(query,normalize_embeddings=True)
        scores = np.dot(self.embeddings, query_embedding)
        results= np.argsort(scores)[::-1][:k]
        for index in results :
            print(f'Similarity:{scores[index]:.4f} Document: {self.documents[index]}')

documents = [
    "Redis is an in-memory data store used for caching.",
    "PostgreSQL is a relational database.",
    "React is a JavaScript library for building user interfaces.",
    "Caching reduces repeated database queries.",
    "FastAPI is a Python framework for building APIs.",
    "Vector databases store numerical embeddings."
]

embeddings = Embeddings()
embeddings.embed(documents)
query = input("Enter your query:")
embeddings.search(query, k=3)
