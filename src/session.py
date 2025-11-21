import json
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages.base import BaseMessage
from typing import List, Dict

# TODO: Store the session store in a persistent database instead of file
class SessionStore: 
    """ Class to store chat sessions with file management """
    def __init__(self, storage_path: str = "chat_sessions.json"): 
        self.storage_path: Path = Path(storage_path)
        self.store: Dict[str, BaseChatMessageHistory] = self._load_from_disk()

    def _load_from_disk(self) -> Dict[str, BaseChatMessageHistory]:
        if not self.storage_path.exists():
            return {}
        with self.storage_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                store = {}
                for session_id, messages_data in data.items():
                    history = InMemoryChatMessageHistory()
                    for msg_data in messages_data:
                        history.add_message(BaseMessage.parse_obj(msg_data))
                    store[session_id] = history
                return store
            except json.JSONDecodeError:
                return {} 

    def _save_to_disk(self):
        data = {}
        for session_id, history in self.store.items():
            data[session_id] = [msg.dict() for msg in history.messages]
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def new_session(self, session_id: str):
        self.store[session_id] = InMemoryChatMessageHistory()
        self._save_to_disk()
        return session_id

    def delete_session(self, session_id: str):
        if session_id in self.store:
            self.store.pop(session_id, None)
            self._save_to_disk()

    def list_sessions(self) -> List[str]:
        return list(self.store.keys())