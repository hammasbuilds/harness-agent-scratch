"""The agent loop, driven by a scripted model. The tools really run."""

import json

import pytest

from harness.agent import STRIPPED_KEEP
from harness.llm import Reply, ScriptedLLM, call
from harness.skills import Skill


def test_final_answer_without_tools(make_agent):
    llm = ScriptedLLM([Reply("hello")])
    agent = make_agent(llm)
    assert agent.send("hi") == "hello"
    assert agent.messages == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_tool_result_is_fed_back_and_the_loop_continues(make_agent, workspace):
    (workspace / "llm.py").write_text("print('hi')\n")
    llm = ScriptedLLM([Reply("", [call("read_file", path="llm.py")]), Reply("It prints hi.")])
    agent = make_agent(llm)
    assert agent.send("what does llm.py do?") == "It prints hi."
    second_request = llm.requests[1][0]
    tool_msg = second_request[-2]  # the last message is the late-injected reminder
    assert tool_msg["role"] == "tool" and tool_msg["content"] == "print('hi')\n"
    assert tool_msg["tool_call_id"] == second_request[-3]["tool_calls"][0]["id"]


def test_several_calls_in_one_reply_all_get_results(make_agent, workspace):
    llm = ScriptedLLM([
        Reply("", [call("write_file", path="a.txt", content="1"), call("write_file", path="b.txt", content="2")]),
        Reply("done"),
    ])
    agent = make_agent(llm)
    agent.send("make two files")
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs] == ["wrote 1 characters to a.txt", "wrote 1 characters to b.txt"]
    assert (workspace / "b.txt").read_text() == "2"


def test_errors_go_back_to_the_model_instead_of_crashing(make_agent):
    llm = ScriptedLLM([
        Reply("", [call("no_such_tool")]),
        Reply("", [call("read_file", path="missing.txt")]),
        Reply("recovered"),
    ])
    agent = make_agent(llm)
    assert agent.send("go") == "recovered"
    results = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert results[0].startswith("error: unknown tool") and results[1].startswith("error: missing.txt")


def test_the_request_prefix_never_changes_within_a_turn(make_agent, workspace):
    """The prefix-cache rule: every request starts with the previous one, minus
    only the reminder at the end."""
    (workspace / "f.txt").write_text("x")
    llm = ScriptedLLM([
        Reply("", [call("write_todos", todos=[{"content": "step", "status": "in_progress"}])]),
        Reply("", [call("read_file", path="f.txt")]),
        Reply("", [call("bash", command="ls")]),
        Reply("done"),
    ])
    agent = make_agent(llm)
    agent.send("do it")
    requests = [r for r, _ in llm.requests]
    for prev, nxt in zip(requests, requests[1:]):
        assert nxt[:len(prev) - 1] == prev[:-1]
    tools = [t for _, t in llm.requests]
    assert all(t == tools[0] for t in tools)


def test_reminder_is_last_and_never_stored(make_agent):
    llm = ScriptedLLM([Reply("", [call("write_todos", todos=[{"content": "plan", "status": "in_progress"}])]),
                       Reply("ok")])
    agent = make_agent(llm)
    agent.send("hi")
    first, second = llm.requests[0][0], llm.requests[1][0]
    assert first[-1]["content"].startswith("<system-reminder>") and "Today: 2026-09-26" in first[-1]["content"]
    assert "[>] plan" not in first[-1]["content"]
    assert "[>] plan" in second[-1]["content"]  # the todo list shows up in the next request
    assert not any("<system-reminder>" in (m.get("content") or "") for m in agent.messages)


def test_system_prompt_is_static_and_lists_skills(cfg, make_agent, tmp_path):
    skill_dir = tmp_path / "skills" / "wordle"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: wordle\ndescription: Solve today's Wordle.\n---\nbody")
    cfg.skills_dirs = [tmp_path / "skills"]
    agent = make_agent(ScriptedLLM([]))
    system = agent.system_message()["content"]
    assert "- wordle: Solve today's Wordle." in system
    assert "body" not in system
    assert "2026" not in system  # dates belong in the reminder, not the prefix


def test_stale_file_warning_reaches_the_model(make_agent, workspace):
    f = workspace / "config.py"
    f.write_text("A = 1\n")

    def edit_behind_its_back(messages):
        f.write_text("A = 2\n")
        import os
        later = f.stat().st_mtime + 5
        os.utime(f, (later, later))
        return Reply("", [call("bash", command="pwd")])

    llm = ScriptedLLM([Reply("", [call("read_file", path="config.py")]), edit_behind_its_back, Reply("ok")])
    make_agent(llm).send("check config")
    assert "- config.py" in llm.requests[2][0][-1]["content"]


def test_old_tool_outputs_are_trimmed_when_a_new_turn_starts(make_agent, workspace):
    (workspace / "big.txt").write_text("y" * 1000)
    llm = ScriptedLLM([Reply("", [call("read_file", path="big.txt")]), Reply("read it"), Reply("second answer")])
    agent = make_agent(llm)
    agent.send("read big.txt")
    assert agent.messages[2]["content"] == "y" * 1000  # still whole during its own turn
    agent.send("next question")
    trimmed = agent.messages[2]["content"]
    assert trimmed.startswith("y" * STRIPPED_KEEP) and "removed after its turn ended" in trimmed
    assert len(trimmed) < 300
    agent._trim_old_tool_outputs()  # a second pass must not re-trim the note
    assert agent.messages[2]["content"] == trimmed


def test_max_steps_stops_a_runaway_loop(cfg, make_agent):
    cfg.max_steps = 3
    llm = ScriptedLLM([Reply("", [call("bash", command="pwd")]) for _ in range(3)])
    assert make_agent(llm).send("loop forever") == "[stopped after 3 steps without a final answer]"


def test_compaction_kicks_in_and_moves_history_into_the_system_prompt(cfg, make_agent, workspace):
    cfg.context_limit = 2000
    (workspace / "big.txt").write_text("z" * 2500)
    llm = ScriptedLLM([
        Reply("", [call("read_file", path="big.txt")]),
        Reply("first done"),
        Reply("HANDOFF: read big.txt, it is all z"),  # the compaction call
        Reply("second done"),
    ])
    events = []
    agent = make_agent(llm, on_event=lambda k, d: events.append(k))
    agent.send("read big.txt")
    # Trimming at the start of the next turn is not enough here: the new question
    # alone is large, so compaction has to run.
    agent.send("now summarise " + "q" * 5000)
    assert "compacted" in events
    assert agent.summary == "HANDOFF: read big.txt, it is all z"
    assert "HANDOFF" in agent.system_message()["content"]
    assert agent.messages[0]["role"] == "user" and agent.messages[0]["content"].startswith("now summarise")
    compaction_request = llm.requests[2]
    assert compaction_request[1] is None  # no tools offered to the summariser


def test_spill_files_are_removed_when_the_turn_ends(cfg, make_agent, workspace):
    cfg.output_cap = 50
    (workspace / "long.txt").write_text("L" * 500)
    llm = ScriptedLLM([Reply("", [call("read_file", path="long.txt")]), Reply("ok")])
    agent = make_agent(llm)
    agent.send("read it")
    assert ".harness/spill/output-1.txt" in agent.messages[2]["content"]
    assert not (workspace / ".harness" / "spill").exists()


def test_llm_errors_propagate_but_the_spill_is_still_cleaned(cfg, make_agent, workspace):
    from harness.llm import LLMError
    cfg.output_cap = 10
    (workspace / "long.txt").write_text("L" * 100)
    llm = ScriptedLLM([Reply("", [call("read_file", path="long.txt")])])  # then runs out
    agent = make_agent(llm)
    with pytest.raises(LLMError):
        agent.send("read it")
    assert not (workspace / ".harness" / "spill").exists()


def test_subagent_is_isolated_and_read_only(make_agent, workspace):
    (workspace / "README.md").write_text("# Neural Code\nA coding harness.")
    llm = ScriptedLLM([
        Reply("", [call("task", prompt="What is this repo?")]),               # main agent delegates
        Reply("", [call("write_file", path="hack.txt", content="x")]),         # subagent tries to write
        Reply("", [call("read_file", path="README.md")]),                      # subagent reads
        Reply("It is Neural Code, a coding harness."),                         # subagent answers
        Reply("The subagent says it is a coding harness."),                    # main agent answers
    ])
    agent = make_agent(llm)
    assert agent.send("explore this repo") == "The subagent says it is a coding harness."

    sub_first, sub_tools = llm.requests[1]
    assert [m["role"] for m in sub_first] == ["system", "user"]  # none of the main conversation
    assert sub_first[1]["content"] == "What is this repo?"
    assert sorted(t["function"]["name"] for t in sub_tools) == ["bash", "read_file", "read_skill"]
    assert "not available to subagents" in llm.requests[2][0][-1]["content"]
    assert not (workspace / "hack.txt").exists()

    main_msgs = agent.messages
    assert [m["role"] for m in main_msgs] == ["user", "assistant", "tool", "assistant"]
    assert main_msgs[2]["content"] == "It is Neural Code, a coding harness."  # only the final answer


def test_read_skill_through_the_loop(cfg, make_agent, tmp_path):
    skill_dir = tmp_path / "skills" / "boids"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: boids\ndescription: Flocking sims.\n---\nUse separation, alignment, cohesion.")
    cfg.skills_dirs = [tmp_path / "skills"]
    llm = ScriptedLLM([Reply("", [call("read_skill", name="boids")]), Reply("ok")])
    agent = make_agent(llm)
    agent.send("make boids")
    assert agent.messages[2]["content"].endswith("Use separation, alignment, cohesion.")


def test_reset_clears_the_session(make_agent):
    llm = ScriptedLLM([Reply("", [call("write_todos", todos=[{"content": "a", "status": "pending"}])]), Reply("x")])
    agent = make_agent(llm)
    agent.send("go")
    agent.summary = "old"
    agent.reset()
    assert agent.messages == [] and agent.summary == "" and agent.todos.items == []


def test_assistant_text_alongside_tool_calls_is_kept(make_agent):
    llm = ScriptedLLM([Reply("Let me look.", [call("bash", command="pwd")]), Reply("done")])
    events = []
    agent = make_agent(llm, on_event=lambda k, d: events.append((k, d)))
    agent.send("go")
    assert ("assistant_text", "Let me look.") in events
    assert agent.messages[1]["content"] == "Let me look."
    assert json.loads(agent.messages[1]["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd"}
