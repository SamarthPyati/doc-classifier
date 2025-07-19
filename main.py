from config import RAGConfig
from system import RAGSystem

import warnings
warnings.filterwarnings('ignore', category=Warning, module='unstructured')
warnings.filterwarnings('ignore', category=UserWarning, module='resource_tracker')      # TODO: Understand and fix semaphore leaks
warnings.filterwarnings('ignore', category=DeprecationWarning)

def main(): 
    config = RAGConfig()
    rag_system = RAGSystem(config)
    rag_system.build_knowledge_base()

    queries: list[str] = [
        "Provide a summary of Single Tender Enquiry (STE) and the auditing process that happens in IFCI.",
        "Provide a summary of procurement policay of Shimla Jal Prabandhan Nigam Limited (SJPNL).",
        "Explain Nature and Scope & Human Resource Planning of HR document of UTKAL University."
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 50)
        
        result = rag_system.query(query)
        print(result)

if __name__ == "__main__": 
    main()


# TODO: System is painfully slow, diagnose the reasons and fix it
# TODO: Enable multiprocessing and mulithreading