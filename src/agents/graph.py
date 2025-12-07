"""LangGraph workflow wiring all agents together."""

from langgraph.graph import END, StateGraph

from .arxiv import arxiv_node
from .parser import parser_node
from .rating import rating_node
from .retriever import retrieve_node
from .reviewer import reviewer_node
from .state import AgentState


def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("parse", parser_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("arxiv", arxiv_node)
    workflow.add_node("review", reviewer_node)
    workflow.add_node("rate", rating_node)

    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "retrieve")
    workflow.add_edge("retrieve", "arxiv")
    workflow.add_edge("arxiv", "review")
    workflow.add_edge("review", "rate")
    workflow.add_edge("rate", END)
    return workflow.compile()


if __name__ == "__main__":
    app = build_agent_graph()
    inputs = AgentState(
        paper_text="This paper proposes a new method for efficient transformer training using low-rank adaptation...",
        paper_title="Efficient Training with LoRA",
    )
    print("Starting Multi-Agent Workflow...")
    for output in app.stream(inputs):
        for key in output:
            print(f"Finished Node: {key}")
    print("Workflow finished.")
