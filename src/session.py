import json
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import messages_from_dict, messages_to_dict
from typing import List, Dict

# TODO: Store the session store in a persistent database instead of file
class SessionStore: 
    """ Class to store chat sessions with file management """
    def __init__(self, storage_path: str = "chat_sessions.json"): 
        self.storage_path: Path = Path(storage_path)
        self.store: Dict[str, BaseChatMessageHistory] = {}
        self.titles: Dict[str, str] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.storage_path.exists():
            return
        
        with self.storage_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Handle legacy format where data was just {session_id: messages}
                # New format: {"sessions": {id: messages}, "titles": {id: title}}
                
                if "sessions" in data and "titles" in data:
                    sessions_data = data["sessions"]
                    self.titles = data["titles"]
                else:
                    sessions_data = data
                    self.titles = {}

                for session_id, messages_data in sessions_data.items():
                    history = InMemoryChatMessageHistory()
                    try:
                        messages = messages_from_dict(messages_data)
                        for msg in messages:
                            history.add_message(msg)
                    except Exception as e:
                        # Fallback or skip corrupted history
                        pass
                    self.store[session_id] = history
                    
                    # Set default title if missing
                    if session_id not in self.titles:
                        self.titles[session_id] = f"Session {session_id[:8]}"
                        
            except json.JSONDecodeError:
                pass

    def save(self):
        sessions_data = {}
        for session_id, history in self.store.items():
            # Use messages_to_dict for proper serialization
            sessions_data[session_id] = messages_to_dict(history.messages)
            
        data = {
            "sessions": sessions_data,
            "titles": self.titles
        }
        
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
            self.titles[session_id] = "New Chat"
        return self.store[session_id]
    
    def new_session(self, session_id: str):
        self.store[session_id] = InMemoryChatMessageHistory()
        self.titles[session_id] = "New Chat"
        self.save()
        return session_id

    def delete_session(self, session_id: str):
        if session_id in self.store:
            self.store.pop(session_id, None)
            self.titles.pop(session_id, None)
            self.save()

    def list_sessions(self) -> List[str]:
        # Return sorted by creation (insertion order in dict is preserved in Python 3.7+)
        # But for robustness, we might want to reverse it to show newest first
        return list(reversed(list(self.store.keys())))

    def set_session_title(self, session_id: str, title: str):
        self.titles[session_id] = title
        self.save()

    def get_session_title(self, session_id: str) -> str:
        return self.titles.get(session_id, f"Session {session_id[:8]}")