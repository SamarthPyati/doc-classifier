from config import RAGConfig
from system import RAGSystem

import warnings
warnings.filterwarnings('ignore', category=Warning, module='unstructured')
warnings.filterwarnings('ignore', category=DeprecationWarning)

def main(): 
    config = RAGConfig()
    rag_system = RAGSystem(config)
    rag_system.build_knowledge_base()

    queries: list[str] = [
        "Provide a summary of procurement policay of Shimla Jal Prabandhan Nigam Limited (SJPNL).",
        "Provide a summary of Single Tender Enquiry (STE) without a PAC and the auditing process that happens in IFCI",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 50)
        
        result = rag_system.query(query)
        print(result)

if __name__ == "__main__": 
    main()