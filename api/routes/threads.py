from fastapi import APIRouter, status
from langgraph.checkpoint.postgres import PostgresSaver
from main import checkpointer, compiled_graph

router = APIRouter()

_is_postgres = isinstance(checkpointer, PostgresSaver)
_LIST_THREADS_QUERY = f"""
    SELECT thread_id, MAX(checkpoint_id) AS latest
    FROM checkpoints
    GROUP BY thread_id
    ORDER BY latest DESC
    LIMIT {"%s" if _is_postgres else "?"}
"""


@router.get("/threads", status_code=status.HTTP_200_OK)
def list_threads(limit: int = 50):
    """
    Lists known conversation threads, most recently active first, each with
    a short preview of its first message - for a sidebar/history view.
    """
    with checkpointer.lock:
        if _is_postgres:
            with checkpointer.conn.cursor() as cur:
                cur.execute(_LIST_THREADS_QUERY, (limit,))
                rows = cur.fetchall()
            thread_ids = [row["thread_id"] for row in rows]
        else:
            rows = checkpointer.conn.execute(_LIST_THREADS_QUERY, (limit,)).fetchall()
            thread_ids = [row[0] for row in rows]

    threads = []
    for thread_id in thread_ids:
        state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", []) if state.values else []
        preview = messages[0].content if messages else "(empty conversation)"
        threads.append({"thread_id": thread_id, "preview": preview[:80]})

    return {"threads": threads}


@router.get("/threads/{thread_id}/messages", status_code=status.HTTP_200_OK)
def get_thread_messages(thread_id: str):
    """
    Returns the saved conversation history for a thread, so a client can
    restore it (e.g. after a page reload) instead of starting a blank chat.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = compiled_graph.get_state(config)
    messages = state.values.get("messages", []) if state.values else []

    return {
        "messages": [
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content,
            }
            for message in messages
        ]
    }
