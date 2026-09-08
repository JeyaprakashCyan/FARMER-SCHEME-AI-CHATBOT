"""
Conversation Memory
===================

Step 12.21

Provides lightweight short-term conversation memory
for the Farmer Scheme Chatbot.

Responsibilities:
    - Store user questions
    - Store assistant answers
    - Retrieve recent conversation history
    - Support multiple sessions
    - Clear a session
    - Format history for the LLM

Important:
    Conversation memory is separate from FAISS.

    FAISS
        -> stores farmer-scheme knowledge

    Memory
        -> stores the current conversation
"""


from typing import Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

# Number of previous conversation turns to remember.
#
# Example:
#
# User question
# Assistant answer
# User question
# Assistant answer
#
# MAX_TURNS = 5
# means the latest 5 user/assistant exchanges are retained.
#
MAX_TURNS = 5


# ============================================================
# CONVERSATION MEMORY CLASS
# ============================================================

class ConversationMemory:
    """
    Lightweight in-memory conversation storage.

    Each session has its own conversation history.

    Example:

        session_1:
            [
                {
                    "role": "user",
                    "content": "What is PM-KISAN?"
                },
                {
                    "role": "assistant",
                    "content": "PM-KISAN provides..."
                }
            ]

        session_2:
            [
                ...
            ]
    """

    def __init__(
        self,
        max_turns: int = MAX_TURNS,
    ):
        self.max_turns = max_turns

        # Dictionary containing all active sessions.
        #
        # Structure:
        #
        # {
        #     "session_id": [
        #         {
        #             "role": "user",
        #             "content": "..."
        #         },
        #         {
        #             "role": "assistant",
        #             "content": "..."
        #         }
        #     ]
        # }
        #
        self.sessions: Dict[
            str,
            List[Dict[str, str]]
        ] = {}

    # ========================================================
    # CREATE / GET SESSION
    # ========================================================

    def _ensure_session(
        self,
        session_id: str,
    ):

        if session_id not in self.sessions:

            self.sessions[session_id] = []

    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    def add_user_message(
        self,
        session_id: str,
        message: str,
    ):

        self._ensure_session(session_id)

        self.sessions[session_id].append(
            {
                "role": "user",
                "content": message.strip(),
            }
        )

        self._trim_history(session_id)

    # ========================================================
    # ADD ASSISTANT MESSAGE
    # ========================================================

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
    ):

        self._ensure_session(session_id)

        self.sessions[session_id].append(
            {
                "role": "assistant",
                "content": message.strip(),
            }
        )

        self._trim_history(session_id)

    # ========================================================
    # ADD COMPLETE TURN
    # ========================================================

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ):

        self._ensure_session(session_id)

        self.sessions[session_id].append(
            {
                "role": "user",
                "content": user_message.strip(),
            }
        )

        self.sessions[session_id].append(
            {
                "role": "assistant",
                "content": assistant_message.strip(),
            }
        )

        self._trim_history(session_id)

    # ========================================================
    # TRIM HISTORY
    # ========================================================

    def _trim_history(
        self,
        session_id: str,
    ):

        history = self.sessions.get(
            session_id,
            [],
        )

        # One turn = user + assistant.
        max_messages = self.max_turns * 2

        if len(history) > max_messages:

            self.sessions[session_id] = (
                history[-max_messages:]
            )

    # ========================================================
    # GET HISTORY
    # ========================================================

    def get_history(
        self,
        session_id: str,
    ) -> List[Dict[str, str]]:

        self._ensure_session(session_id)

        return list(
            self.sessions[session_id]
        )

    # ========================================================
    # GET RECENT HISTORY
    # ========================================================

    def get_recent_history(
        self,
        session_id: str,
        turns: Optional[int] = None,
    ) -> List[Dict[str, str]]:

        self._ensure_session(session_id)

        if turns is None:

            turns = self.max_turns

        max_messages = turns * 2

        return list(
            self.sessions[session_id][
                -max_messages:
            ]
        )

    # ========================================================
    # FORMAT HISTORY
    # ========================================================

    def format_history(
        self,
        session_id: str,
        turns: Optional[int] = None,
    ) -> str:

        history = self.get_recent_history(
            session_id=session_id,
            turns=turns,
        )

        if not history:

            return "No previous conversation."

        formatted_lines = []

        for message in history:

            role = message.get(
                "role",
                "unknown",
            )

            content = message.get(
                "content",
                "",
            )

            if role == "user":

                formatted_lines.append(
                    f"User: {content}"
                )

            elif role == "assistant":

                formatted_lines.append(
                    f"Assistant: {content}"
                )

        return "\n".join(
            formatted_lines
        )

    # ========================================================
    # CHECK WHETHER SESSION HAS HISTORY
    # ========================================================

    def has_history(
        self,
        session_id: str,
    ) -> bool:

        return bool(
            self.sessions.get(
                session_id,
                [],
            )
        )

    # ========================================================
    # CLEAR SESSION
    # ========================================================

    def clear_session(
        self,
        session_id: str,
    ):

        if session_id in self.sessions:

            del self.sessions[session_id]

    # ========================================================
    # CLEAR ALL SESSIONS
    # ========================================================

    def clear_all(self):

        self.sessions.clear()

    # ========================================================
    # SESSION COUNT
    # ========================================================

    def session_count(self) -> int:

        return len(
            self.sessions
        )

    # ========================================================
    # PRINT MEMORY
    # ========================================================

    def print_memory(
        self,
        session_id: str,
    ):

        print("\n")
        print("=" * 60)
        print("CONVERSATION MEMORY")
        print("=" * 60)

        history = self.get_history(
            session_id
        )

        if not history:

            print("No conversation history.")

            return

        for i, message in enumerate(
            history,
            start=1,
        ):

            role = message.get(
                "role",
                "unknown",
            )

            content = message.get(
                "content",
                "",
            )

            print(
                f"\n[{i}] {role.upper()}"
            )

            print(content)

        print("\n" + "=" * 60)


# ============================================================
# SIMPLE TEST
# ============================================================

def main():

    print("=" * 60)
    print("CONVERSATION MEMORY TEST")
    print("=" * 60)

    memory = ConversationMemory(
        max_turns=3
    )

    session_id = "test-session"

    # --------------------------------------------------------
    # Add first conversation turn
    # --------------------------------------------------------

    memory.add_turn(
        session_id=session_id,
        user_message=(
            "What is Pradhan Mantri "
            "Kisan Samman Nidhi?"
        ),
        assistant_message=(
            "PM-KISAN is a Central Sector "
            "Scheme that provides income "
            "support to landholding "
            "farmers' families."
        ),
    )

    # --------------------------------------------------------
    # Add second conversation turn
    # --------------------------------------------------------

    memory.add_turn(
        session_id=session_id,
        user_message=(
            "What are its benefits?"
        ),
        assistant_message=(
            "The scheme provides financial "
            "support to eligible farmers."
        ),
    )

    # --------------------------------------------------------
    # Print history
    # --------------------------------------------------------

    memory.print_memory(
        session_id
    )

    # --------------------------------------------------------
    # Test formatted history
    # --------------------------------------------------------

    print("\nFORMATTED HISTORY")
    print("=" * 60)

    print(
        memory.format_history(
            session_id
        )
    )

    # --------------------------------------------------------
    # Test session count
    # --------------------------------------------------------

    print("\n")
    print(
        f"Active sessions: "
        f"{memory.session_count()}"
    )

    # --------------------------------------------------------
    # Test clear
    # --------------------------------------------------------

    memory.clear_session(
        session_id
    )

    print(
        f"History after clear: "
        f"{len(memory.get_history(session_id))}"
    )

    print("\n✓ MEMORY TEST PASSED")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

