from src import RAGConfig
from argparse import ArgumentParser
from src import RAGSystem

def parser(config: RAGConfig):
    parser = ArgumentParser(
        prog="Document Classifier",
        description="A RAG Pipeline to retrieve relevant documents for LLM interaction.",
        usage="python main.py <command> [options]", 
        add_help=True
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # query <query>
    query_parser = subparsers.add_parser("query", help="Ask a question using the RAG system")
    query_parser.add_argument("query", type=str, help="The query string")

    # index [corpus_path] [--overwrite]
    index_parser = subparsers.add_parser("index", help="Index documents from a folder")
    index_parser.add_argument(
        "corpus_path",
        nargs="?",
        default=config.corpus_path,
        type=str,
        help="Path to document corpus (default: 'corpus')"
    )
    index_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index")

    # chat [--session, --stream]
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--session", type=str, help="Specific session ID to use")
    chat_parser.add_argument("--stream", action="store_true", help="Enable streaming responses")

    # Chat session management 
    session_parser = subparsers.add_parser("sessions", help="Manage chat sessions")
    session_parser.add_argument("--list", action="store_true", help="List all sessions")
    session_parser.add_argument("--clear", type=str, help="Clear specific session")
    session_parser.add_argument("--history", type=str, help="Show history for session")

    return parser.parse_args() 