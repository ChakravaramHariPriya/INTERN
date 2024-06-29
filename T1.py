import openai
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch

pc = Pinecone(
    api_key='your_pinecone_api_key'
)

index_name = 'your_index_name'
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

index = pc.index(index_name)

openai.api_key = 'your_openai_api_key'

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings[0].cpu().numpy()

def retrieve_documents(query, top_k=5):
    query_embedding = embed_text(query)
    results = index.query(query_embedding, top_k=top_k, include_values=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_answer(query):
    documents = retrieve_documents(query)
    
    prompt = f"Question: {query}\n\nContext:\n" + "\n\n".join(documents) + "\n\nAnswer:"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    query = "What are the business hours of XYZ company?"
    answer = generate_answer(query)
    print(f"Query: {query}\nAnswer: {answer}")
