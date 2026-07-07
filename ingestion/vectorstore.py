"""Vector store and embedding setup for MultiDocChat."""


def get_embedding_function(provider="local"):
    """Return the configured embedding function.

    Keep embedding provider selection centralized here so the project can
    switch from local embeddings to OpenAI embeddings later without touching
    retrieval code.
    """
    if provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    raise ValueError(f"Unsupported embedding provider: {provider}")
