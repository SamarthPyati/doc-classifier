from src import RAGConfig
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parser(config: RAGConfig):
    # Use a formatter that shows default values in help messages 
    def formatter(prog): 
        return ArgumentDefaultsHelpFormatter(prog, max_help_position=35)
    
    parser = ArgumentParser(
        prog="Document Classifier",
        description="A RAG Pipeline to retrieve relevant documents for LLM interaction.",
        usage="python main.py <command> [options]", 
        formatter_class=formatter, 
        add_help=True
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # --- Index Command ---
    index_parser = subparsers.add_parser("index", help="Build or check the knowledge base from documents.", formatter_class=formatter)
    index_parser.add_argument(
        "corpus_path",
        nargs="?",
        default=config.corpus_path,
        type=str,
        help="Path to the document corpus directory."
    )
    index_parser.add_argument("--overwrite", action="store_true", help="Force rebuild, overwriting the existing index.")
    index_parser.add_argument("--check", action="store_true", help="Check the number of documents currently in the index.")

    # --- Query Command ---
    query_parser = subparsers.add_parser("query", help="Ask a single question and get a direct answer.", formatter_class=formatter)
    query_parser.add_argument("query", type=str, help="The question to ask the RAG system.")

    # --- Chat Command ---
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.", formatter_class=formatter)
    chat_parser.add_argument("--session", type=str, default=None, help="A specific session ID to resume a previous chat.")
    chat_parser.add_argument("--stream", action="store_true", help="Enable real-time streaming for responses.")

    # --- Sessions Command ---
    session_parser = subparsers.add_parser("session", help="Manage chat sessions.", formatter_class=formatter)
    session_group = session_parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--list", action="store_true", help="List all saved session IDs.")
    session_group.add_argument("--history", type=str, metavar="SESSION_ID", help="Show the conversation history for a specific session.")
    session_group.add_argument("--clear", type=str, metavar="SESSION_ID", help="Clear the history for a specific session.")

    # --- Database Subcommand ---
    db_parser = subparsers.add_parser("db", help="Manage database.", formatter_class=formatter)
    db_group = db_parser.add_mutually_exclusive_group(required=True)
    db_group.add_argument("--peek", type=int, help="Show first 'n' documents in database.", default=10)
    db_group.add_argument("--count", action="store_true", help="Returns the count of entires in database.")

    # --- Pseudo Test Subcommand ---
    db_parser = subparsers.add_parser("test", help="Test by re-indexing the database and answering 3 prompts.", formatter_class=formatter)

    return parser.parse_args()