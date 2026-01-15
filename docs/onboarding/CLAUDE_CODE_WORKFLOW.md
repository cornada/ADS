# Claude Code workflow (project default)

**Generated:** 2026-01-14

## How we use Claude Code here
- Claude Code is the **terminal agent**: it edits files, runs tests, and reports outputs.
- The project contract is in `CLAUDE.md` (must follow).

## Suggested prompt skeleton
> Read docs/issues/<KT file>.  
> Propose a plan (files/tests/commands).  
> Implement.  
> Run pytest -q and the acceptance commands.  
> Report outputs + paths + commit message.

## Debugging rule
When something fails:
1) paste the full stack trace,
2) ask for a minimal fix,
3) rerun the exact failing command.
