import sys
import warnings
import logging

from src.config import setup_logging
from src import RAGConfig, RAGSystem
from parser import parser

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=Warning, module='unstructured')
warnings.filterwarnings('ignore', category=UserWarning, module='resource_tracker')      # TODO: Understand and fix semaphore leaks
warnings.filterwarnings('ignore', category=DeprecationWarning)

ENABLE_PARSER: bool = True

def main(): 
    setup_logging()
    config = RAGConfig()    

    try: 
        if ENABLE_PARSER: 
            args = parser(config)
            rag_system = RAGSystem(config)

            if args.command == "index":
                if args.check: 
                    print(f"📚 Number of documents indexed in database: {rag_system.document_count()}")
                else: 
                    config.corpus_path = args.corpus_path
                    print(f"📚 Indexing documents from: {config.corpus_path}")
                    success = rag_system.build_knowledge_base(force_rebuild=args.overwrite)
                    if success:
                        print("✅ Knowledge base built successfully!")
                    else:
                        print("❌ Failed to build knowledge base.")

            elif args.command == "query":
                print(f"\n❓ Query: {args.query}")
                print("=" * 100)
                result = rag_system.query(args.query)
                print(f"🤖 Response: {result.response}")
                if result.sources:
                    print(f"\n📚 Sources: {', '.join(result.sources)}")
                    print(f"🎯 Confidence: {result.confidence:.2f}")
                    print(f"⚡ Time: {result.processing_time:.2f}s")

            elif args.command == "chat":
                interactive_chat(rag_system, args.session, args.stream)

            elif args.command == "sessions":
                manage_sessions(rag_system, args)
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
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def interactive_chat(rag_system: RAGSystem, session_id: str = None, enable_streaming: bool = False):
    """Enhanced interactive chat with LangChain"""
    
    if session_id:
        rag_system.current_session_id = session_id
        print(f"📱 Using session: {session_id}")
    else:
        session_id = rag_system.create_new_session()
        print(f"📱 Created new session: {session_id}")

    print("\n" + "="*80)
    print("🦜 LangChain RAG Chat System - Interactive Mode")
    print("="*80)
    print("Commands:")
    print("  /help      - Show this help")
    print("  /history   - Show conversation history")
    print("  /clear     - Clear conversation history")
    print("  /new       - Start new chat session")
    print("  /sessions  - List all sessions")
    print("  /stream    - Toggle streaming mode")
    print("  /quit      - Exit chat")
    print("="*80)
    
    streaming_enabled = enable_streaming
    
    while True:
        try:
            user_input = input(f"\n🧑 You [{session_id[:8]}...]: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit':
                    print("👋 Goodbye!")
                    break
                elif command == '/help':
                    print("\nAvailable commands:")
                    print("  /help      - Show this help")
                    print("  /history   - Show conversation history")
                    print("  /clear     - Clear conversation history")
                    print("  /new       - Start new chat session")
                    print("  /sessions  - List all sessions")
                    print("  /stream    - Toggle streaming mode")
                    print("  /quit      - Exit chat")
                elif command == '/history':
                    history = rag_system.get_chat_history(session_id)
                    if not history:
                        print("📝 No conversation history yet.")
                    else:
                        print(f"\n📝 Chat History ({len(history)} messages):")
                        print("-" * 50)
                        for msg in history:
                            role_emoji = "🧑" if msg["role"] == "human" else "🤖"
                            print(f"{role_emoji} {msg['role']}: {msg['content'][:100]}...")
                elif command == '/clear':
                    rag_system.clear_chat_history(session_id)
                    print("🗑️ Chat history cleared.")
                elif command == '/new':
                    session_id = rag_system.create_new_session()
                    print(f"🆕 New chat session started: {session_id}")
                elif command == '/sessions':
                    sessions = rag_system.list_sessions()
                    print(f"📋 Active sessions ({len(sessions)}):")
                    for sid in sessions:
                        marker = "👈 current" if sid == session_id else ""
                        print(f"  {sid[:8]}... {marker}")
                elif command == '/stream':
                    streaming_enabled = not streaming_enabled
                    print(f"🔄 Streaming mode: {'enabled' if streaming_enabled else 'disabled'}")
                else:
                    print("❓ Unknown command. Type /help for available commands.")
                continue
            
            # Process chat message
            if streaming_enabled:
                print("🤖 Assistant: ", end="", flush=True)
                full_response = ""
                for chunk in rag_system.stream_chat(user_input, session_id):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # New line after streaming
                
                # Note: We'd need to manually track sources in streaming mode
                # For now, get them with a regular call for metadata
                result = rag_system.chat(user_input, session_id)
                if result.sources:
                    print(f"\n📚 Sources ({result.num_sources}): {', '.join(result.sources)}")
                    print(f"🎯 Confidence: {result.confidence:.2f}")
                    print(f"⚡ Time: {result.processing_time:.2f}s")
            else:
                result = rag_system.chat(user_input, session_id)
                print(f"🤖 Assistant: {result.response}")
                
                if result.sources:
                    print(f"\n📚 Sources ({result.num_sources}): {', '.join(result.sources)}")
                    print(f"🎯 Confidence: {result.confidence:.2f}")
                    print(f"⚡ Time: {result.processing_time:.2f}s")
                
        except KeyboardInterrupt:
            print("\n👋 Chat session ended.")
            break
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            print(f"❌ Error: {e}")

def manage_sessions(rag_system: RAGSystem, args):
    """Manage chat sessions"""
    if args.list:
        sessions = rag_system.list_sessions()
        print(f"📋 Active sessions ({len(sessions)}):")
        for sid in sessions:
            print(f"  {sid}")
    
    elif args.clear:
        rag_system.clear_chat_history(args.clear)
        print(f"🗑️ Cleared session: {args.clear}")
    
    elif args.history:
        history = rag_system.get_chat_history(args.history)
        if not history:
            print("📝 No conversation history.")
        else:
            print(f"📝 Chat History for {args.history} ({len(history)} messages):")
            for msg in history:
                role_emoji = "🧑" if msg["role"] == "human" else "🤖"
                print(f"{role_emoji} {msg['role']}: {msg['content']}")

if __name__ == "__main__": 
    main()


# TODO: Indexing is painfully slow, diagnose the problem and fix it. 
# TODO: Enable multiprocessing and mulithreading.