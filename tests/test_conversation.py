"""Tests for ConversationState."""

import pytest
from textualgrok.conversation import ConversationState


def make_state(prompt="You are helpful."):
    return ConversationState(system_prompt=prompt)


class TestBuildRequest:
    def test_minimal_no_history(self):
        state = make_state()
        messages = state.build_request("Hello")
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[-1] == {"role": "user", "content": "Hello"}
        assert len(messages) == 2

    def test_with_history(self):
        state = make_state()
        state.commit_turn("Hi", "Hey there!")
        messages = state.build_request("Follow-up")
        # system + 2 history + new user = 4
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["content"] == "Follow-up"

    def test_exclude_history(self):
        state = make_state()
        state.commit_turn("Hi", "Hey there!")
        messages = state.build_request("Fresh", include_history=False)
        assert len(messages) == 2
        assert messages[-1]["content"] == "Fresh"

    def test_user_content_parts_appended(self):
        state = make_state()
        parts = [{"type": "input_image", "image_url": "http://img"}]
        messages = state.build_request("Describe this", user_content_parts=parts)
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == parts[0]
        assert user_msg["content"][-1] == {"type": "input_text", "text": "Describe this"}

    def test_system_prompt_always_first(self):
        state = make_state("Be concise.")
        state.commit_turn("q", "a")
        messages = state.build_request("new")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."


class TestCommitTurn:
    def test_appends_user_and_assistant(self):
        state = make_state()
        state.commit_turn("What time is it?", "It is noon.")
        assert len(state.history) == 2
        assert state.history[0] == {"role": "user", "content": "What time is it?"}
        assert state.history[1] == {"role": "assistant", "content": "It is noon."}

    def test_multiple_turns(self):
        state = make_state()
        state.commit_turn("a", "b")
        state.commit_turn("c", "d")
        assert len(state.history) == 4

    def test_accepts_non_string_user_content(self):
        state = make_state()
        content = [{"type": "input_image", "image_url": "http://x"}]
        state.commit_turn(content, "Got it.")
        assert state.history[0]["content"] is content


class TestClearHistory:
    def test_clears_all(self):
        state = make_state()
        state.commit_turn("x", "y")
        state.clear_history()
        assert state.history == []

    def test_clear_empty_is_fine(self):
        state = make_state()
        state.clear_history()
        assert state.history == []
