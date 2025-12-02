import os
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===== 설정 =====
AWS_REGION = "us-east-1"
S3_BUCKET = "your-bucket-name"
S3_PREFIX = "vectorstore/"
USE_S3 = False  # True: S3에서 로드, False: 로컬

print("🤖 AWS RAG 챗봇 초기화\n")

# ===== 1. S3에서 벡터 스토어 다운로드 (옵션) =====
if USE_S3:
    print("☁️  S3에서 벡터 스토어 다운로드 중...")
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    os.makedirs("vectorstore", exist_ok=True)
    
    files = ['index.faiss', 'index.pkl']
    for file_name in files:
        s3_key = f"{S3_PREFIX}{file_name}"
        local_path = f"vectorstore/{file_name}"
        
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        print(f"   ✅ {file_name}")
    
    print("   완료\n")

# ===== 2. Embeddings 초기화 =====
print("🔢 Embeddings 초기화...")
try:
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=AWS_REGION
    )
    print("   ✅ Bedrock Titan Embeddings\n")
except Exception as e:
    print(f"   ⚠️ Bedrock 실패, OpenAI로 폴백\n")
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

# ===== 3. FAISS 로드 (Dense Vector) =====
print("📊 FAISS 벡터 스토어 로드...")
vectorstore = FAISS.load_local(
    "vectorstore", 
    embeddings, 
    allow_dangerous_deserialization=True
)
faiss_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
print("   ✅ Dense Vector Retriever\n")

# ===== 4. BM25 Retriever (Sparse Vector) =====
print("📝 BM25 Retriever 구성...")

# 원본 청크 재구성 (BM25용)
local_pdfs = [
    "./docs/wellarchitected-machine-learning-lens.pdf",
]

all_docs = []
for pdf in local_pdfs:
    if os.path.exists(pdf):
        loader = PyPDFLoader(pdf)
        all_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(all_docs)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5
print(f"   ✅ Sparse Vector Retriever ({len(chunks)}개 청크)\n")

# ===== 5. Hybrid Retriever (RRF) =====
print("🔗 Hybrid Search 구성...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]  # 동일 가중치
)
print("   ✅ RRF (Reciprocal Rank Fusion)\n")

# ===== 6. LLM 초기화 =====
print("🧠 LLM 초기화...")
try:
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name=AWS_REGION,
        model_kwargs={"temperature": 0.1}
    )
    print("   ✅ Bedrock Claude 3.5 Sonnet\n")
except Exception as e:
    print(f"   ⚠️ Bedrock 실패, OpenAI로 폴백\n")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

# ===== 7. RAG Chain =====
print("⚙️  RAG Chain 구성...\n")

template = """You are an expert on AWS Well-Architected Framework.

**Instructions:**
1. Answer based ONLY on the provided context
2. Structure answers with bullet points
3. Include specific AWS services mentioned
4. Cite sources at the end

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,  # Hybrid!
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("✅ 초기화 완료!\n")

# ===== 8. 질문 함수 =====
def ask_question(query):
    """질문-답변 함수"""
    print(f"\n{'='*70}")
    print(f"❓ 질문: {query}")
    print(f"{'='*70}\n")
    print("🔍 Hybrid Search 수행 중...\n")
    
    result = qa_chain.invoke({"query": query})
    answer = result['result']
    source_docs = result['source_documents']
    
    print(f"💡 답변:\n{answer}\n")
    
    # 출처 정리
    print(f"📚 참조 출처 ({len(source_docs)}개):")
    sources = {}
    for doc in source_docs:
        source = doc.metadata.get('source', 'Unknown')
        doc_name = doc.metadata.get('doc_name', source.split('/')[-1])
        page = doc.metadata.get('page', 'N/A')
        
        if doc_name not in sources:
            sources[doc_name] = []
        sources[doc_name].append(page)
    
    for idx, (doc_name, pages) in enumerate(sources.items(), 1):
        pages_str = ', '.join(map(str, sorted(set(pages))[:3]))
        print(f"  [{idx}] {doc_name} (페이지: {pages_str})")
    
    return answer

# ===== Main =====
if __name__ == "__main__":
    print("="*70)
    print("🤖 AWS Well-Architected Chatbot")
    print("   - Hybrid Search (BM25 + FAISS)")
    print("   - Bedrock Claude 3.5 + Titan Embeddings")
    print("="*70)
    
    # 테스트 질문
    test_questions = [
        "What are security best practices for ML models?",
        "How to optimize costs in generative AI?",
    ]
    
    print("\n🧪 테스트 모드:\n")
    for q in test_questions:
        ask_question(q)
        print("\n")
    
    # 대화형 모드
    print("="*70)
    print("💬 대화 모드 (종료: 'quit')")
    print("="*70)
    
    while True:
        query = input("\n🧑 질문: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 종료합니다.")
            break
        if query:
            ask_question(query)