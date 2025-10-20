class BaseNode:
    """
    베이스 노드 클래스

    Args:
        chat_model: 챗 모델 인스턴스
    """

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def get_message(self, state, idx: int = -1):
        """특정 인덱스의 메시지 조회"""
        messages = state.get("messages", [])
        if (
            messages
            and type(messages) is list
            and len(messages) > idx
            and hasattr(messages[idx], "content")
        ):
            return messages[idx].content
        return ""
