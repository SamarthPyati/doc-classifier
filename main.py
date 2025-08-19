#!/opt/homebrew/Caskroom/miniconda/base/envs/kpmg/bin/python3
import sys
import asyncio
from typing import Any

from src.config import setup_logging
from src import RAGConfig, RAGSystem
from parser import parser

import logging
logger = logging.getLogger(__name__)

async def handle_index_command(system: RAGSystem, args: Any) -> None:
    """Handles the 'index' command by running the blocking function in an executor."""
    # TODO: Run indexing in a different thread
    logger.info(f"📚 Indexing documents from: {args.corpus_path}")
    try:
        success = await system.build_knowledge_base(args.multiprocess, args.overwrite)
        if success and args.overwrite:
            print("✅ Knowledge base built successfully!")
        elif success:
            pass
        else:
            print("❌ Failed to build knowledge base.")
    except Exception as e:
        logger.error(f"An error occurred during indexing: {e}", exc_info=True)
        print("❌ An unexpected error occurred during indexing. Check logs for details.")

async def handle_query_command(system: RAGSystem, args: Any) -> None: 
    """Handles the 'query' command asynchronously."""
    print(f"\n❓ Query: {args.query}")
    print("=" * 100)
    print("🤔 Thinking...", end="\r", flush=True)
    result = await system.query(args.query)
    sys.stdout.write("\033[K")  # Clear the "Thinking..." line
    print(f"🤖 Response: {result.response}")
    if result.sources:
        print(f"\n📚 Sources: {', '.join(result.sources)}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"⚡ Time: {result.processing_time:.2f}s")

def print_chat_help(header: bool = False) -> None: 
    if header: 
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

async def interactive_chat(rag_system: RAGSystem, session_id: str | None = None, enable_streaming: bool = False):
    """ Enhanced interactive chat with LangChain """

    # Load the chats from the disk
    rag_system.session_store._load_from_disk()

    if session_id:
        rag_system.current_session_id = session_id
        print(f"📱 Using session: {session_id}")
    else:
        session_id = rag_system.create_new_session()
        print(f"📱 Created new session: {session_id}")

    print_chat_help(header=True)
    
    streaming_enabled = enable_streaming
    
    while True:
        try:
            user_input = await asyncio.to_thread(input, f"🧑 You [{session_id[:8]}...]: ")
            
            if not user_input:  
                continue

            user_input = user_input.strip()
                
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit':
                    # Save the session
                    rag_system.session_store._save_to_disk()
                    break

                elif command == '/help':
                    print_chat_help(header=False)

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
                async for chunk in rag_system.stream_chat(user_input, session_id):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                # NOTE: We'd need to manually track sources in streaming mode
                # For now, get them with a regular call for metadata
                result = await rag_system.chat(user_input, session_id)
                if result.sources:
                    print(result)
            else:
                result = await rag_system.chat(user_input, session_id)
                print(f"🤖 Assistant: {result.response}")
                
                if result.sources:
                    print(result)
                
        except KeyboardInterrupt:
            print("\n👋 Chat session ended.")
            break
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            print(f"❌ Error: {e}")

async def handle_chat_command(system: RAGSystem, args: Any) -> None: 
    return await interactive_chat(system, args.session, args.stream)

def handle_session_command(system: RAGSystem, args: Any) -> None:
    """ Manage chat sessions """
    if args.list:
        sessions = system.list_sessions()
        print(f"📋 Active sessions ({len(sessions)}):")
        for sid in sessions:
            print(f"  {sid}")
    
    elif args.clear:
        system.clear_chat_history(args.clear)
        print(f"🗑️ Cleared session: {args.clear}")
    
    elif args.history:
        history = system.get_chat_history(args.history)
        if not history:
            print("📝 No conversation history.")
        else:
            print(f"📝 Chat History for {args.history} ({len(history)} messages):")
            for msg in history:
                role_emoji = "🧑" if msg["role"] == "human" else "🤖"
                print(f"{role_emoji} {msg['role']}: {msg['content']}")

def handle_db_command(system: RAGSystem, args: Any) -> None: 
    if args.count:
        print(f"📚 Number of documents indexed in database: {system.document_count()}")
    elif args.peek: 
        system.list_document(args.peek)

async def handle_test_command(system: RAGSystem) -> None:
    """ Pseudo Test code to test the system by rebuilding and answering 3 prewritten prompts """
    await system.build_knowledge_base()

    queries: list[str] = [
        "Provide a summary of Single Tender Enquiry (STE) and the auditing process that happens in IFCI.",
        "Provide a summary of procurement policay of Shimla Jal Prabandhan Nigam Limited (SJPNL).",
        "Explain Nature and Scope & Human Resource Planning of HR document of UTKAL University."
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 100)

        result = await system.query(query)
        print(result)

async def main(): 
    setup_logging()
    config = RAGConfig()    

    args = parser(config)
    rag_system = RAGSystem(config)

    try:
        match args.command:
            case "index":
                await handle_index_command(rag_system, args)
            case "query":
                await handle_query_command(rag_system, args)
            case "chat":
                await handle_chat_command(rag_system,args)
            case "test":
                await handle_test_command(rag_system)
            case "session":
                handle_session_command(rag_system, args)
            case "db":
                handle_db_command(rag_system, args)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Graceful Shutdown
        logger.info("Shutting down application and background tasks.")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__": 
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
        sys.exit(0)