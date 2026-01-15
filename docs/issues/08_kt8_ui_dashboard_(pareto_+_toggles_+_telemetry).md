# [KT8] UI dashboard (Pareto + toggles + telemetry)

## Why
A usable dashboard strengthens Human-Centred AI positioning: show Pareto, allow toggles, log interactions for study.

## Scope
### In scope
- Minimal dashboard page with Pareto scatter and option list
- Controls: objective selection, tau constraint
- Telemetry logging endpoint + event schema

### Out of scope
- Pixel-perfect design; production auth

## Deliverables
- [ ] UI app skeleton (React/Next or reused scaffold) wired to API
- [ ] Telemetry events stored (file/DB) for sessions
- [ ] Demo script: end-to-end scenario

## Acceptance criteria (must run and paste outputs/paths in PR)
- [ ] `pytest -q`
- [ ] `uvicorn ads_api.main:app --reload  # API runs`
- [ ] `manual UI run: show Pareto and update on toggles`

## Expected artifacts (examples)
- `UI screenshots/gif in docs/`
- `telemetry log file (if implemented)`

## Implementation notes
- If you reuse a scaffold (e.g., HypoCompass), verify license and keep as submodule/fork.

## Prompt for Cursor Composer
> Implement a minimal UI that calls the API to show Pareto results and supports toggles. Log telemetry events.

## Prompt for Claude Code
> Add a minimal UI scaffold (or placeholder) and ensure API endpoints support toggles and telemetry logging.
