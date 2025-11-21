import json
import os
import hashlib
from typing import List, Dict

import chromadb
from chromadb.config import Settings


def build_vectordb_from_json(json_file: str, collection_name: str = "iclr_2025_reviews"):
    """
    JSON 파일에서 논문 초록과 리뷰를 읽어 ChromaDB Vector Database를 구축합니다.
    
    :param json_file: get_all_conference_reviews로 생성된 JSON 파일 경로
    :param collection_name: ChromaDB 컬렉션 이름
    """
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} papers")
    
    # ChromaDB 클라이언트 초기화 (영구 저장)
    db_path = "./chromadb"
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)
    
    # 기존 컬렉션이 있으면 삭제하고 새로 생성
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "ICLR 2025 paper abstracts and reviews for RAG"}
    )
    
    # 데이터를 Vector DB에 추가
    documents = []
    metadatas = []
    ids = []
    
    total_reviews = 0
    for paper in data:
        forum_id = paper['forum_id']
        abstract = paper['abstract']
        
        # 초록을 문서로 추가
        doc_abstract = f"Abstract: {abstract}"
        documents.append(doc_abstract)
        metadatas.append({
            'type': 'abstract',
            'forum_id': forum_id,
        })
        ids.append(f"abs_{forum_id}")
        
        # 각 리뷰를 별도 문서로 추가
        for review in paper['reviews']:
            note_id = review['note_id']
            review_content = review['content']
            
            doc_review = f"Review: {review_content}"
            documents.append(doc_review)
            metadatas.append({
                'type': 'review',
                'forum_id': forum_id,
                'note_id': note_id,
            })
            ids.append(f"rev_{note_id}")
            total_reviews += 1
    
    print(f"Adding {len(documents)} documents to ChromaDB ({total_reviews} reviews)...")
    
    # 배치 크기를 제한하여 추가 (ChromaDB 제약)
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"Added batch {i//batch_size + 1}: {len(batch_docs)} documents")
    
    print(f"\nVector DB 구축 완료!")
    print(f"Collection: {collection_name}")
    print(f"Total documents: {collection.count()}")
    print(f"DB path: {db_path}")
    
    return collection


def test_vectordb_query(collection_name: str = "iclr_2025_reviews", query: str = None):
    """
    ChromaDB에서 유사도 검색 테스트
    """
    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_collection(name=collection_name)
    
    if query is None:
        query = "What are the weaknesses of this paper regarding experimental validation?"
    
    print(f"\nQuery: {query}")
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print("\nTop 3 similar documents:")
    for i, (doc, meta, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n--- Result {i+1} (distance: {distance:.4f}) ---")
        print(f"Type: {meta.get('type')}")
        print(f"Forum ID: {meta.get('forum_id')}")
        print(f"Content: {doc[:300]}...")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ChromaDB from OpenReview data")
    parser.add_argument('--json_file', type=str, 
                       default='data/ICLR.cc_2025_Conference_reviews.json',
                       help='JSON file with reviews')
    parser.add_argument('--collection', type=str, 
                       default='iclr_2025_reviews',
                       help='ChromaDB collection name')
    parser.add_argument('--test', action='store_true',
                       help='Test the vector DB with a sample query')
    parser.add_argument('--query', type=str,
                       help='Custom query for testing')
    
    args = parser.parse_args()
    
    if args.test:
        test_vectordb_query(args.collection, args.query)
    else:
        if not os.path.exists(args.json_file):
            print(f"Error: {args.json_file} not found")
            print("Please run data_collection.py with --batch flag first")
        else:
            build_vectordb_from_json(args.json_file, args.collection)
