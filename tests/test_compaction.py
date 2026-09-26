import pytest

from harness.compaction import (COMPACT_SYSTEM, choose_cut, compact, estimate_tokens, fit_tokens,
                                render_for_summary, text_tokens)
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


# Real Qwen2.5 token counts for these exact 3,000-character samples, measured
# with its tokenizer.json (tokenisation only). The estimate must never be lower.
QWEN_COUNTS = [
    ("\n".join(str(i) for i in range(1, 100001))[:3000], 3000),  # seq: every digit is a token
    ("\n".join(f"{i * 7919 % 100000:05d},{i * 104729 % 1000:03d}.{i % 97:02d}" for i in range(300))[:3000], 3000),
    (("-rw-r--r-- 1 dell 197121  48213 Sep 26 12:07 file_1.py\n" * 60)[:3000], 2074),
]


@pytest.mark.parametrize("text, real", QWEN_COUNTS)
def test_estimate_is_pessimistic_for_numbers(text, real):
    # Characters/3 put these at a third of their real size, so three CSV reads
    # silently overflowed an 8k Ollama context.
    assert text_tokens(text) >= real


def test_estimate_is_pessimistic_for_random_strings():
    import base64
    import os
    text = base64.b64encode(os.urandom(2250)).decode()  # Qwen: about 2,230 tokens for these 3,000 characters
    assert text_tokens(text) >= 2230


@pytest.mark.parametrize("alphabet", ["ACGT", "ACDEFGHIKLMNPQRSTVWY", "abcdefghijklmnopqrstuvwxyz",
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"])
def test_estimate_is_pessimistic_for_single_case_random_letters(alphabet):
    # DNA, protein, lowercase noise and base32 measured 0.51-0.60 Qwen tokens per
    # character; the estimate was 0.29 when only case switches marked randomness.
    import random
    rng = random.Random(7)
    text = "".join(rng.choice(alphabet) for _ in range(3000))
    assert text_tokens(text) >= 0.6 * len(text)


def test_real_words_are_not_priced_as_random():
    prose = "the agent reads a file, changes one line and runs the tests again. " * 40
    assert text_tokens(prose) < len(prose) / 2  # Qwen gives about one token per 4.5 characters here


def test_estimate_is_pessimistic_for_non_latin_text_too():
    msgs = [{"role": "user", "content": "字" * 4000}]
    assert estimate_tokens(msgs) >= 4000  # tokenizers give about one token per CJK character; chars/3 gave 1,343


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
    text = render_for_summary(turn(1, size=5000))
    assert "USER: task 1" in text
    assert '-> called bash {"command": "ls"}' in text
    assert "TOOL RESULT (bash):" in text and "[...]" in text
    assert len(text) < 2000


def test_render_keeps_the_goal_and_the_end_when_over_budget():
    msgs = [{"role": "user", "content": "GOAL: build boids"}] + turn(1, 3000) + turn(2, 3000) + \
        [{"role": "assistant", "content": "LATEST STEP"}]
    text = fit_tokens(render_for_summary(msgs), 400)
    assert "GOAL: build boids" in text and "LATEST STEP" in text and "omitted" in text
    assert text_tokens(text) <= 400


def test_compact_summarises_the_dropped_part():
    msgs = turn(1) + turn(2) + turn(3)
    llm = ScriptedLLM([Reply("NOTE: tasks 1-2 done")])
    summary, kept = compact(msgs, "", llm, keep_tokens=estimate_tokens(turn(3)) + 5, budget_tokens=20_000)
    assert summary == "NOTE: tasks 1-2 done"
    assert kept == turn(3)
    sent, tools = llm.requests[0]
    assert tools is None
    assert sent[0]["content"].startswith(COMPACT_SYSTEM) and "Keep the note under" in sent[0]["content"]
    assert "task 1" in sent[1]["content"] and "task 3" not in sent[1]["content"]


def test_long_notes_are_capped_so_repeated_compactions_cannot_grow_without_bound():
    llm = ScriptedLLM([Reply("n" * 10_000)])
    note, _ = compact(turn(1) + turn(2), "", llm, keep_tokens=estimate_tokens(turn(2)) + 5,
                      budget_tokens=20_000, max_summary_tokens=300)
    assert text_tokens(note) <= 300 and note.endswith("[note cut to fit the context]")


def test_an_empty_note_keeps_an_excerpt_instead_of_losing_history():
    llm = ScriptedLLM([Reply("")])
    note, kept = compact(turn(1) + turn(2), "OLD", llm, keep_tokens=estimate_tokens(turn(2)) + 5,
                         budget_tokens=20_000)
    assert "OLD" in note and "task 1" in note and "summary failed" in note
    assert kept == turn(2)


def test_compact_carries_the_previous_note_forward():
    llm = ScriptedLLM([Reply("new note")])
    compact(turn(1) + turn(2), "OLD NOTE", llm, keep_tokens=estimate_tokens(turn(2)) + 5, budget_tokens=20_000)
    assert "OLD NOTE" in llm.requests[0][0][1]["content"]


def test_compact_does_nothing_when_nothing_can_be_dropped():
    llm = ScriptedLLM([])
    msgs = [{"role": "user", "content": "x"}]
    assert compact(msgs, "prev", llm, keep_tokens=1, budget_tokens=100) == ("prev", msgs)
    assert llm.requests == []


def test_a_non_latin_note_is_capped_in_tokens_not_characters():
    # A character cap sized for English let a Chinese note take 45% of the context.
    llm = ScriptedLLM([Reply("进度" * 3000)])
    note, _ = compact(turn(1) + turn(2), "", llm, keep_tokens=estimate_tokens(turn(2)) + 5,
                      budget_tokens=20_000, max_summary_tokens=1200)
    assert text_tokens(note) <= 1200


def test_the_transcript_sent_for_summarising_fits_its_budget():
    msgs = [{"role": "user", "content": "目标" * 5000}] + turn(1, 3000) + turn(2)
    llm = ScriptedLLM([Reply("note")])
    compact(msgs, "", llm, keep_tokens=estimate_tokens(turn(2)) + 5, budget_tokens=2000)
    body = llm.requests[0][0][1]["content"]
    assert text_tokens(body) <= 2000


def test_fit_tokens_leaves_short_text_alone_and_keeps_both_ends():
    assert fit_tokens("short", 100) == "short"
    cut = fit_tokens("A" * 3000 + "Z" * 3000, 500)
    assert cut.startswith("A") and cut.endswith("Z") and text_tokens(cut) <= 500
