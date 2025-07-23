import logging
from argparse import ArgumentParser

from config import RAGConfig
from system import RAGSystem

import sys
import warnings
warnings.filterwarnings('ignore', category=Warning, module='unstructured')
warnings.filterwarnings('ignore', category=UserWarning, module='resource_tracker')      # TODO: Understand and fix semaphore leaks
warnings.filterwarnings('ignore', category=DeprecationWarning)

logger = logging.getLogger(__name__)

def parser(config: RAGConfig):
    parser = ArgumentParser(
        prog="Document Classifier",
        description="A RAG Pipeline to retrieve relevant documents for LLM interaction.",
        usage="python main.py <command> [options]"
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

ENABLE_PARSER: bool = False

def main(): 
    config = RAGConfig()    
    rag_system = RAGSystem(config)

    try: 
        if ENABLE_PARSER: 
            args = parser(config)

            if args.command == "index":
                config.corpus_path = args.corpus_path
                rag_system.build_knowledge_base(force_rebuild=args.overwrite)

            elif args.command == "query":
                print(f"\nQuery: {args.query}")
                print("=" * 100)
                result = rag_system.query(args.query)
                print(result)
        else: 
            rag_system = RAGSystem(config)
            rag_system.build_knowledge_base()

            queries: list[str] = [
                "Provide a summary of Single Tender Enquiry (STE) and the auditing process that happens in IFCI.",
                "Provide a summary of procurement policay of Shimla Jal Prabandhan Nigam Limited (SJPNL).",
                "Explain Nature and Scope & Human Resource Planning of HR document of UTKAL University."
            ]

            for query in queries:
                print(f"\nQuery: {query}")
                print("=" * 100)
                
                result = rag_system.query(query)
                print(result)
    except KeyboardInterrupt:
        logger.error("Encountered KeyBoard Interrupt. Exiting program ...")
        sys.exit(-1)

if __name__ == "__main__": 
    main()


# TODO: System is painfully slow, diagnose the reasons and fix it
# TODO: Enable multiprocessing and mulithreading