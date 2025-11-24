'''
Purpose: PDF 로딩 → 텍스트 정제 → Pillar/BP ID 감지 → Chunking → 유효성 검증
'''
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

DOCUMENT_DIR = "./docs"
VECTOR_DB_DIR = "./vectorstore"

# Pillar 정의 (문서 순서대로)
PILLAR_PATTERNS = {
    "Operational Excellence": ["operational excellence", "OPS0", "OPS1"],
    "Security": ["security pillar", "SEC0", "SEC1", "SEC2"],
    "Reliability": ["reliability pillar", "REL0", "REL1"],
    "Performance Efficiency": ["performance efficiency", "PERF0", "PERF1"],
    "Cost Optimization": ["cost optimization", "COST0", "COST1"],
    "Sustainability": ["sustainability pillar", "SUS0", "SUS1"]
}

def detect_pillar(text: str, current_pillar: str = "General") -> str:
    """텍스트에서 Pillar 감지 (더 정교한 버전)"""
    text_lower = text.lower()
    
    for pillar, keywords in PILLAR_PATTERNS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return pillar
    
    return current_pillar  # 감지 안되면 이전 Pillar 유지

def detect_best_practice_id(text: str) -> str:
    """Best Practice ID 감지 (패턴 확장)"""
    # SEC01-BP01, OPS02-BP03, REL01-BP02 등
    patterns = [
        r'([A-Z]{2,4}\d{1,2}-BP\d{1,2})',  # SEC01-BP01
        r'([A-Z]{2,4}\d{1,2}\s*-\s*BP\s*\d{1,2})',  # SEC01 - BP 01 (공백 포함)
        r'(Best practice \d+\.\d+)',  # Best practice 1.1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "").upper()
    
    return ""

def clean_text(text: str) -> str:
    """불필요한 텍스트 정리"""
    # 반복 패턴 제거
    text = re.sub(r'(fault tolerance, high availability,?\s*)+', '', text)
    text = re.sub(r'(reliability pillar, cloud architecture best practices,?\s*)+', '', text)
    text = re.sub(r'(AWS Well-Architected Framework,?\s*)+', '', text, flags=re.IGNORECASE)
    # 헤더/푸터 패턴 제거
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # 페이지 번호만 있는 줄
    # 과도한 공백 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def is_valid_chunk(text: str) -> bool:
    """유효한 청크인지 검증"""
    if len(text) < 100:
        return False
    if text.count("well-architected") > 10:
        return False
    if text.count("...") > 10:  # 목차
        return False
    if text.count("•") > 20 and len(text) < 500:  # 목록만 있는 경우
        return False
    return True

def load_and_clean_documents(path: str):
    """PDF 로딩 + 정제 + Pillar 누적 추적"""
    documents = []
    current_pillar = "General"
    
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(path, filename)
            print(f"📄 Loading: {filename}")
            
            loader = PyPDFLoader(filepath)
            raw_docs = loader.load()
            
            # 페이지 순서대로 처리 (Pillar 누적 추적)
            for doc in sorted(raw_docs, key=lambda x: x.metadata.get("page", 0)):
                cleaned_text = clean_text(doc.page_content)
                
                # Pillar 감지 (현재 페이지에서 새 Pillar 발견하면 업데이트)
                current_pillar = detect_pillar(cleaned_text, current_pillar)
                bp_id = detect_best_practice_id(cleaned_text)
                
                if is_valid_chunk(cleaned_text):
                    enhanced_metadata = {
                        "source": filename,
                        "page": doc.metadata.get("page", 0) + 1,  # 1-indexed
                        "pillar": current_pillar,
                        "best_practice_id": bp_id,
                    }
                    
                    documents.append(Document(
                        page_content=cleaned_text,
                        metadata=enhanced_metadata
                    ))
    
    print(f"✅ 정제 후 문서 수: {len(documents)}")
    return documents

def split_documents(documents):
    """의미 단위 Chunking"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n\n\n",
            "\n\n",
            "\nBest practice",
            "\nRequired:",
            "\nRecommended:",
            "\n• ",
            "\n",
            ". ",
            " "
        ]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # 청크별 BP ID 재감지
    valid_chunks = []
    for chunk in chunks:
        if is_valid_chunk(chunk.page_content):
            # 청크 내에서 BP ID 다시 찾기
            bp_id = detect_best_practice_id(chunk.page_content)
            if bp_id:
                chunk.metadata["best_practice_id"] = bp_id
            valid_chunks.append(chunk)
    
    print(f"✅ 유효 청크 수: {len(valid_chunks)}")
    return valid_chunks


def create_vectorstore(chunks):
    """Embedding 생성 + FAISS 저장"""
    
    print("\n" + "="*50)
    print("🔄 Embedding 생성 시작")
    print("="*50)
    
    # Bedrock Titan Embeddings
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )
    
    print(f"📊 총 {len(chunks)}개 청크 임베딩 중...")
    print("⏳ 예상 소요 시간: 5-10분")
    
    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # 로컬 저장
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_DIR)
    
    print(f"✅ VectorStore 로컬 저장 완료: {VECTOR_DB_DIR}")
    print(f"   - index.faiss")
    print(f"   - index.pkl")
    
    return vectorstore


def test_retrieval(vectorstore):
    """검색 테스트"""
    test_queries = [
        "How should IAM authentication be designed?",
        "What is the tagging strategy for cost optimization?",
        "How should Recovery Time Objective (RTO) and Recovery Point Objective (RPO) be set for disaster recovery?",
        "What are the methods for optimizing Lambda function performance?"
    ]
    
    print("\n" + "="*50)
    print("🔍 검색 테스트")
    print("="*50)
    
    for query in test_queries:
        print(f"\n질문: {query}")
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            pillar = doc.metadata.get('pillar', 'Unknown')
            page = doc.metadata.get('page', '?')
            bp_id = doc.metadata.get('best_practice_id', '-')
            
            print(f"\n  [{i}] {pillar} | Page {page} | BP: {bp_id}")
            print(f"      {doc.page_content[:200]}...")


if __name__ == "__main__":
    # 1. 문서 로딩 + 정제
    docs = load_and_clean_documents(DOCUMENT_DIR)
    
    # 2. Chunking
    chunks = split_documents(docs)
    
    # 3. Pillar 분포 확인
    pillar_counts = {}
    bp_counts = 0
    for chunk in chunks:
        pillar = chunk.metadata.get("pillar", "Unknown")
        pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1
        if chunk.metadata.get("best_practice_id"):
            bp_counts += 1
    
    print("\n" + "="*50)
    print("📊 Pillar 분포")
    print("="*50)
    for pillar, count in sorted(pillar_counts.items(), key=lambda x: -x[1]):
        print(f"  {pillar}: {count}개")
    
    print(f"\n📌 BP ID가 있는 청크: {bp_counts}개 ({bp_counts*100//len(chunks)}%)")
    
    # 4. Embedding + FAISS 저장
    vectorstore = create_vectorstore(chunks)
    
    # 5. 검색 테스트
    test_retrieval(vectorstore)
    
    print("\n" + "="*50)
    print("✅ 완료!")
    print("="*50)
    print(f"다음 단계: AWS 콘솔에서 {VECTOR_DB_DIR}/ 폴더의")
    print("index.faiss와 index.pkl을 S3에 업로드하세요.")
    print(f"S3 경로: s3://korea-sw-16-chatbot-s3/vectorstore/")

# if __name__ == "__main__":
#     docs = load_and_clean_documents(DOCUMENT_DIR)
#     chunks = split_documents(docs)
    
#     # Pillar 분포 확인
#     pillar_counts = {}
#     bp_counts = 0
#     for chunk in chunks:
#         pillar = chunk.metadata.get("pillar", "Unknown")
#         pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1
#         if chunk.metadata.get("best_practice_id"):
#             bp_counts += 1
    
#     print("\n" + "="*50)
#     print("📊 Pillar 분포")
#     print("="*50)
#     for pillar, count in sorted(pillar_counts.items(), key=lambda x: -x[1]):
#         print(f"  {pillar}: {count}개")
    
#     print(f"\n📌 BP ID가 있는 청크: {bp_counts}개 ({bp_counts*100//len(chunks)}%)")
    
#     # 샘플 확인
#     print("\n" + "="*50)
#     print("샘플 청크 확인")
#     print("="*50)
    
#     for i in [0, 100, 500, 1000, 1500]:
#         if i < len(chunks):
#             c = chunks[i]
#             print(f"\n--- 청크 #{i} ---")
#             print(f"Pillar: {c.metadata.get('pillar')}")
#             print(f"BP ID: {c.metadata.get('best_practice_id') or '(없음)'}")
#             print(f"Page: {c.metadata.get('page')}")
#             print(f"Content: {c.page_content[:200]}...")
        