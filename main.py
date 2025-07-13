from config import RAGConfig
from system import RAGSystem

def main(): 
    config = RAGConfig()

    rag_system = RAGSystem(config)

    rag_system.build_knowledge_base()

    queries = [
        "Get me the function for Attaching Shader",
        "Explain in detail about buffer in opengl",
        "Basic Datatypes in OpenGL"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 50)
        
        result = rag_system.query(query)
        
        print(f"Response: {result.response}")
        print(f"Sources: {result.sources}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Number of sources: {result.num_sources}")

if __name__ == "__main__": 
    main()