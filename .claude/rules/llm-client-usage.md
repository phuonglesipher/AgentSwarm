# LLM Client Usage

## Rules

- `LLMClient` for single-turn text/JSON generation (review, planning, routing)
- `ClaudeCodeExecutorClient` (`core/executor.py`) for multi-turn tool-using execution (code changes, investigation)
- Always check `llm.is_enabled()` before calling; return graceful fallback on failure
- Use `ensure_traced_llm_client()` wrapper for automatic prompt/response logging
- `LLMManager.resolve(profile_name)` returns the right client for a named role
- Retry logic is built into `_retry_with_backoff()` — do not add custom retry loops
- Executor has anti-recursion guard — never invoke `main.py` or `claude_bridge.py` from executor prompts
