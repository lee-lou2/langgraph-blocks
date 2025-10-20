# langgraph-blocks

LangGraph 기반 에이전트를 빠르게 조립하기 위한 재사용 블록 라이브러리

## 빠른 시작
```bash
git clone https://github.com/lee-lou2/langgraph-blocks.git
cd langgraph-blocks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
파이썬 3.11 이상을 권장하며 OpenRouter 모델을 쓸 경우 `OPENROUTER_API_KEY` 같은 환경 변수를 미리 준비하세요

## 프로젝트 구조
- `models/`: 채팅·임베딩·리랭크 모델 래퍼
- `nodes/`: 그래프 상태를 변형하는 노드 클래스
- `tools/`: LangChain 호환 도구 모음
- `core/`: 데이터베이스 및 공용 유틸리티
- `prompts/`: 재사용 가능한 프롬프트 템플릿

## 주요 패키지와 클래스
- `models.chat_models.ChatLocal`: 로컬 서버로 호스팅된 OpenAI 호환 엔드포인트를 사용하는 경량 채팅 클래스
- `models.chat_models.ChatOpenRouter`: OpenRouter API와 통신하며 모델 이름과 키만으로 교체 가능한 채팅 클래스
- `models.embedding_models.LocalEmbedding`: Hugging Face 임베딩 모델을 간단히 교체할 수 있는 래퍼
- `models.reranking_models.LocalReranking`: Qwen 기반 리랭클 모델을 호출해 문서 점수를 반환하는 클래스
- `core.databases.Milvus`: 하이브리드 검색을 위한 Milvus 컬렉션 생성과 질의를 관리하는 헬퍼
- `nodes.QueryRewrite`: 입력 메시지를 기반으로 검색 친화적 질문을 재작성하는 LangGraph 노드
- `tools.calculator.calculator`: 안전한 AST 평가로 수식을 계산하는 LangChain 도구
- `tools.http.http_get`: 단순 GET 요청을 수행하고 응답을 반환하는 도구
- `tools.file_system.read_file`: 작업 디렉터리 내 파일을 읽어오는 도구
- `tools.python_repl.python_repl`: 제한된 네임스페이스에서 파이썬 코드를 실행하는 도구

## 사용 예시
```python
from langgraph.graph import StateGraph, END
from models.chat_models import ChatLocal
from nodes import QueryRewrite

graph = StateGraph(dict)

rewrite = QueryRewrite(ChatLocal())
graph.add_node("rewrite", rewrite.as_node)
graph.set_entry_point("rewrite")
graph.add_edge("rewrite", END)

compiled = graph.compile()
result = compiled.invoke({"messages": [...]})
```
원격 모델을 쓰려면 `ChatLocal` 대신 `ChatOpenRouter`를 주입하고 필요한 환경 변수를 설정하세요 새 도구를 추가할 때는 `tools/__init__.py`에서 등록 흐름을 맞춰 주세요

## CLI 워크플로
`main.py`에서 `graph` 객체를 노출했다면 다음 명령으로 라이브 리로드 개발을 진행할 수 있습니다
```bash
langgraph dev main:graph
```
그래프 로직을 바꾸면 CLI가 자동으로 새 구성을 반영합니다
