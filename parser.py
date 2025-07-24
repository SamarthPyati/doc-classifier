from src import RAGConfig
from argparse import ArgumentParser

def parser(config: RAGConfig):
    parser = ArgumentParser(
        prog="Document Classifier",
        description="A RAG Pipeline to retrieve relevant documents for LLM interaction.",
        usage="python main.py <command> [options]", 
        add_help=True
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── query <query>
    query_parser = subparsers.add_parser("query", help="Ask a question using the RAG system")
    query_parser.add_argument("query", type=str, help="The query string")

    # ── index [corpus_path] [--overwrite]
    index_parser = subparsers.add_parser("index", help="Index documents from a folder")
    index_parser.add_argument(
        "corpus_path",
        nargs="?",
        default=config.corpus_path,
        type=str,
        help="Path to document corpus (default: 'corpus')"
    )
    index_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index")

    return parser.parse_args() 