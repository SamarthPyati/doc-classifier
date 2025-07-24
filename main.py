import sys
import warnings
import logging

from src import RAGConfig, RAGSystem
from parser import parser

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=Warning, module='unstructured')
warnings.filterwarnings('ignore', category=UserWarning, module='resource_tracker')      # TODO: Understand and fix semaphore leaks
warnings.filterwarnings('ignore', category=DeprecationWarning)


ENABLE_PARSER: bool = True

def main(): 
    config = RAGConfig()    

    try: 
        if ENABLE_PARSER: 
            args = parser(config)
            rag_system = RAGSystem(config)

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


# TODO: Indexing is painfully slow, diagnose the problem and fix it. 
# TODO: Enable multiprocessing and mulithreading.