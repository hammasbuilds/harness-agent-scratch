from harness.compaction import COMPACT_SYSTEM, choose_cut, compact, estimate_tokens, render_for_summary
from harness.llm import Reply, ScriptedLLM


def turn(n: int, size: int = 300) -> list[dict]:
    return [
        {"role": "user", "content": f"task {n}"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{n}", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "tool_call_id": f"c{n}", "name": "bash", "content": "x" * size},
        {"role": "assistant", "content": f"done {n}"},
    ]


def test_estimate_is_pessimistic():
    msgs = [{"role": "user", "content": "a" * 3000}]
    assert estimate_tokens(msgs) > 3000 / 4  # a real tokenizer gives ~750 here


def test_cut_lands_on_a_user_message_and_fits():
    msgs = turn(1) + turn(2) + turn(3) + turn(4)
    keep = estimate_tokens(turn(4)) + 5
    cut = choose_cut(msgs, keep)
    assert msgs[cut] == {"role": "user", "content": "task 4"}
    assert estimate_tokens(msgs[cut:]) <= keep


def test_the_current_turn_is_never_cut():
    # One huge turn and nothing older: compaction must not summarise away the
    # user's own request (it once did, cutting at the first assistant message).
    msgs = [{"role": "user", "content": "go"}]
    for n in range(6):
        msgs += turn(n)[1:3]
    assert choose_cut(msgs, keep_tokens=200) == 0


def test_when_the_current_turn_alone_is_too_big_all_older_turns_go():
    msgs = turn(1) + turn(2) + [{"role": "user", "content": "q" * 3000}]
    cut = choose_cut(msgs, keep_tokens=100)
    assert msgs[cut]["content"] == "q" * 3000
    assert all(m["role"] != "tool" or any(tc["id"] == m["tool_call_id"] for p in msgs[cut:]
                                          for tc in p.get("tool_calls") or []) for m in msgs[cut:])


def test_cut_zero_when_nothing_can_go():
    assert choose_cut([{"role": "user", "content": "only"}], keep_tokens=1) == 0


def test_render_includes_calls_and_trims_tool_output():
    text = render_for_summary(turn(1, size=5000), budget_chars=100_000)
    assert "USER: task 1" in text
    assert '-> called bash {"command": "ls"}' in text
    assert "TOOL RESULT (bash):" in text and "[...]" in text
    assert len(text) < 2000


def test_render_keeps_the_goal_and_the_end_when_over_budget():
    msgs = [{"role": "user", "content": "GOAL: build boids"}] + turn(1, 3000) + turn(2, 3000) + \
        [{"role": "assistant", "content": "LATEST STEP"}]
    text = render_for_summary(msgs, budget_chars=1200)
    assert "GOAL: build boids" in text and "LATEST STEP" in text and "omitted" in text


def test_compact_summarises_the_dropped_part():
    msgs = turn(1) + turn(2) + turn(3)
    llm = ScriptedLLM([Reply("NOTE: tasks 1-2 done")])
    summary, kept = compact(msgs, "", llm, keep_tokens=estimate_tokens(turn(3)) + 5, budget_chars=50_000)
    assert summary == "NOTE: tasks 1-2 done"
    assert kept == turn(3)
    sent, tools = llm.requests[0]
    assert tools is None
    assert sent[0]["content"].startswith(COMPACT_SYSTEM) and "Keep the note under" in sent[0]["content"]
    assert "task 1" in sent[1]["content"] and "task 3" not in sent[1]["content"]


def test_long_notes_are_capped_so_repeated_compactions_cannot_grow_without_bound():
    llm = ScriptedLLM([Reply("n" * 10_000)])
    note, _ = compact(turn(1) + turn(2), "", llm, keep_tokens=estimate_tokens(turn(2)) + 5,
                      budget_chars=50_000, max_summary_chars=1000)
    assert len(note) < 1100 and note.endswith("[note cut to fit the context]")


def test_an_empty_note_keeps_an_excerpt_instead_of_losing_history():
    llm = ScriptedLLM([Reply("")])
    note, kept = compact(turn(1) + turn(2), "OLD", llm, keep_tokens=estimate_tokens(turn(2)) + 5,
                         budget_chars=50_000)
    assert "OLD" in note and "task 1" in note and "summary failed" in note
    assert kept == turn(2)


def test_compact_carries_the_previous_note_forward():
    llm = ScriptedLLM([Reply("new note")])
    compact(turn(1) + turn(2), "OLD NOTE", llm, keep_tokens=estimate_tokens(turn(2)) + 5, budget_chars=50_000)
    assert "OLD NOTE" in llm.requests[0][0][1]["content"]


def test_compact_does_nothing_when_nothing_can_be_dropped():
    llm = ScriptedLLM([])
    msgs = [{"role": "user", "content": "x"}]
    assert compact(msgs, "prev", llm, keep_tokens=1, budget_chars=100) == ("prev", msgs)
    assert llm.requests == []
