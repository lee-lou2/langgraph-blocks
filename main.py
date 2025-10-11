from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict
from fastapi import FastAPI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from llms.openrouter.base import ChatOpenRouter


class GraphState(TypedDict):
    question: str
    role_play: str
    response: str


def set_role_node(state):
    question = state.get("question")
    role = "당신은 세계 최고의 요리사입니다. 사용자의 질문에 레시피를 알려주세요."
    return {"role_play": role, "question": question}


def generate_response_node(state):
    question = state.get("question")
    role_play = state.get("role_play")
    llm = ChatOpenRouter(model="x-ai/grok-4-fast")
    prompt = f"{role_play}\n\n사용자 질문: {question}"
    response = llm.invoke(prompt)
    return {"response": response.content}


workflow = StateGraph(GraphState)
workflow.add_node("set_role", set_role_node)
workflow.add_node("generate_response", generate_response_node)

workflow.set_entry_point("set_role")
workflow.add_edge("set_role", "generate_response")
workflow.add_edge("generate_response", END)

app_runnable = workflow.compile()

api = FastAPI(
    title="LangGraph Recipe Server",
    version="1.0",
    description="A simple API server using LangGraph to provide recipes",
)


# 요청 바디를 위한 Pydantic 모델 정의
class Request(BaseModel):
    input: str
    conversation_id: str  # 세션 관리를 위한 ID


# 응답 바디를 위한 Pydantic 모델 정의
class Response(BaseModel):
    output: str


@api.post("/invoke", response_model=Response)
async def invoke_agent(request: Request):
    config = {"configurable": {"thread_id": request.conversation_id}}
    result = app_runnable.invoke({"messages": [("user", request.input)]}, config)
    final_output = result["response"]
    return Response(output=final_output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
