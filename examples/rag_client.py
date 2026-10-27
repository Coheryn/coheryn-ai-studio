#!/usr/bin/env python3
"""Optional LangChain + local Qdrant example against Coheryn's embeddings API.

Install tools/requirements-rag.txt in an isolated environment. The core Rust
engine does not depend on these clients or store documents/vectors itself.
"""
import argparse
import json
import os


def demo(base_url, model, token):
    import httpx
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client import QdrantClient, models

    identity = []
    def pin_space(response):
        if response.is_success:
            current = response.headers.get('x-coheryn-embedding-space')
            if not current or identity and current != identity[0]:
                raise ValueError('embedding-space identity missing or changed; rebuild the index explicitly')
            if not identity:
                identity.append(current)
    transport = httpx.Client(trust_env=False, event_hooks={'response': [pin_space]})
    embeddings = OpenAIEmbeddings(
        base_url=base_url, api_key=token, model=model,
        check_embedding_ctx_length=False,  # Send text; Coheryn owns tokenization.
        chunk_size=8, max_retries=0, http_client=transport,
    )
    documents = [
        'search_document: Local inference runs models on your own computer.',
        'search_document: Bananas grow in tropical climates.',
    ]
    vectors = embeddings.embed_documents(documents)
    database = QdrantClient(location=':memory:')
    try:
        database.create_collection('coheryn', vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE))
        database.upsert('coheryn', points=[models.PointStruct(
            id=index, vector=vector, payload={'text': text, 'embedding_space': identity[0]})
            for index, (text, vector) in enumerate(zip(documents, vectors, strict=True))])
        query = embeddings.embed_query('search_query: Where does local inference run?')
        matches = database.query_points('coheryn', query=query, limit=2).points
        return {'embedding_space': identity[0], 'dimension': len(query), 'documents': len(vectors),
                'matches': [{'id': p.id, 'score': p.score, 'text': p.payload['text']} for p in matches]}
    finally:
        database.close()
        transport.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:11450/v1')
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    token = os.environ.get('COHERYN_API_TOKEN')
    if not token:
        parser.error('set COHERYN_API_TOKEN to a Coheryn inference credential')
    print(json.dumps(demo(args.base_url, args.model, token), indent=2))
