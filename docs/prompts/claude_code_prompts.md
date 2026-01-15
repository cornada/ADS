# Prompt library — Claude Code

Use these snippets inside `claude` sessions (terminal agent).
Always include acceptance criteria.

## Global pattern
> Read docs/issues/<KT>.md.  
> Propose a plan (files, tests, commands).  
> Implement.  
> Run tests and smoke.  
> Report outputs and paths.

## Debugging pattern
> I got an error: <paste stack trace>.  
> Diagnose and propose fix.  
> Implement minimal change.  
> Run pytest and the failing command again.

## Refactor pattern
> Refactor <module> to <new structure> without changing behavior.  
> Add/update tests to ensure behavior unchanged.

## Reproducibility pattern
> Ensure every run writes config_resolved.yaml, seed, model_id, and dataset hashes.  
> Add a small unit test verifying the files exist.
