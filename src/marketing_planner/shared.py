"""Shared objects across the application."""

from langgraph.store.memory import InMemoryStore

# Create memory store
memory_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)