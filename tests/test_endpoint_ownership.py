"""
Unit tests for the pattern-B ownership-check helpers added alongside the
auth layer: topic_quiz.py and topic_tending.py accept only a `session_id`
(no `user_id`), so ownership can only be enforced by loading the row and
comparing its `user_id` to the verified caller. These monkeypatch the DB
layer directly — no live Supabase calls.
"""
import pytest
from fastapi import HTTPException

import app.api.v1.endpoints.topic_quiz as topic_quiz_ep
import app.api.v1.endpoints.topic_tending as topic_tending_ep


class FakeResp:
    def __init__(self, data):
        self.data = data


@pytest.mark.asyncio
async def test_assert_session_owner_matches(monkeypatch):
    async def fake_select(*args, **kwargs):
        return FakeResp({"user_id": "user-a"})

    monkeypatch.setattr(topic_quiz_ep, "db_select_single", fake_select)
    await topic_quiz_ep._assert_session_owner("session-1", "user-a")


@pytest.mark.asyncio
async def test_assert_session_owner_mismatch_is_403(monkeypatch):
    async def fake_select(*args, **kwargs):
        return FakeResp({"user_id": "user-a"})

    monkeypatch.setattr(topic_quiz_ep, "db_select_single", fake_select)
    with pytest.raises(HTTPException) as exc:
        await topic_quiz_ep._assert_session_owner("session-1", "user-b")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_assert_session_owner_missing_is_404(monkeypatch):
    async def fake_select(*args, **kwargs):
        return FakeResp(None)

    monkeypatch.setattr(topic_quiz_ep, "db_select_single", fake_select)
    with pytest.raises(HTTPException) as exc:
        await topic_quiz_ep._assert_session_owner("session-1", "user-a")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_load_session_for_owner_matches(monkeypatch):
    async def fake_run(op):
        return FakeResp({"id": "session-1", "user_id": "user-a"})

    monkeypatch.setattr(topic_tending_ep, "run_db_operation", fake_run)
    session = await topic_tending_ep._load_session_for_owner("session-1", "user-a")
    assert session["user_id"] == "user-a"


@pytest.mark.asyncio
async def test_load_session_for_owner_mismatch_is_403(monkeypatch):
    async def fake_run(op):
        return FakeResp({"id": "session-1", "user_id": "user-a"})

    monkeypatch.setattr(topic_tending_ep, "run_db_operation", fake_run)
    with pytest.raises(HTTPException) as exc:
        await topic_tending_ep._load_session_for_owner("session-1", "user-b")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_load_session_for_owner_missing_is_404(monkeypatch):
    async def fake_run(op):
        return FakeResp(None)

    monkeypatch.setattr(topic_tending_ep, "run_db_operation", fake_run)
    with pytest.raises(HTTPException) as exc:
        await topic_tending_ep._load_session_for_owner("session-1", "user-a")
    assert exc.value.status_code == 404
