# Contributing

## Before You Open a PR

- Keep changes focused on one concern.
- Do not commit local `.env` files, local domains, private IPs, tokens, or logs.
- Update docs when behavior, configuration, or API messages change.
- Run the smoke tests locally before pushing:

```bash
STT_SKIP_MODEL_LOAD=1 python3 -m unittest discover -s tests -p 'test_*.py'
```

## Pull Request Expectations

- Describe what changed and why.
- Include verification details.
- Call out config or deployment impact explicitly.
- If AI assistance was used, say so clearly and confirm you reviewed the code.

## Dependency and License Expectations

- Prefer permissive dependencies that are compatible with the MIT license used by this repository.
- If you add a new dependency, document why it is needed.

## Deployment Safety

- Treat `cache/` and `tmp/` as runtime state, not source files.
- Use `scripts/sync-remote.sh` for remote syncs so bind-mounted runtime data is not deleted.