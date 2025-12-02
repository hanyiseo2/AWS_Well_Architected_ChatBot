'''
from langchain_core import hub
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool

# (이하 코드 생략)

# 1. 도구 정의 (데코레이터 사용)
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 2. 모델 정의 (API 키가 환경 변수로 설정되어 있어야 합니다)
# **주의**: 모델 이름은 현재 시점의 Anthropic 모델 이름으로 변경했습니다.
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# 3. 프롬프트 가져오기 (ReAct 에이전트 기본 프롬프트 사용)
# 이 프롬프트는 에이전트에게 도구 사용 방법을 안내합니다.
prompt = hub.pull("hwchase17/react")

# 4. 에이전트 생성
# create_react_agent는 LLM, Tools, Prompt를 받아서 Runnable 객체를 반환합니다.
agent = create_react_agent(llm, [get_weather], prompt)

# 5. AgentExecutor를 사용하여 에이전트 실행 환경 생성
# AgentExecutor는 에이전트와 도구를 연결하여 실행 루프를 관리합니다.
agent_executor = AgentExecutor(agent=agent, tools=[get_weather], verbose=True)

# 6. 에이전트 실행
agent_executor.invoke(
    {"input": "what is the weather in sf"} # 'messages' 대신 'input' 사용
)
'''
from typing import Union

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "healthy"}