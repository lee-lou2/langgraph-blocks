from langchain_core.messages import HumanMessage

from nodes.base import BaseNode


class QueryRewrite(BaseNode):
    """쿼리 재작성 노드"""

    PROMPT = """Look at the input and try to reason about the underlying semantic intent / meaning. \n 
Here is the initial question:
\n ------- \n
%s 
\n ------- \n
Formulate an improved question:"""

    def as_node(self, state):
        """
        쿼리 재작성 노드

        Args:
            state: 현재 상태

        Returns:
            재작성된 쿼리
        """
        first_message = self.get_message(state, idx=0)
        if not first_message:
            return {"query": ""}
        assert self.chat_model is not None, "Model is not set"
        response = self.chat_model.invoke(
            [HumanMessage(content=self.PROMPT % first_message)]
        )
        return {"rewritten_query": response.content}
