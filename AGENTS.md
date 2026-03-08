## Repository Guardrails

- Before any `git add`, `git commit`, `git tag`, `git push`, or `npm publish`, audit the effective shipped content, not just the edited diff.
- Scan git-visible files for local/private data such as loopback or LAN IPs, localhost endpoints, home-directory paths, local clone paths, tokens, secrets, passwords, auth headers, machine-specific config, and private env var names or values.
- If this repo has `package.json`, run `npm pack --dry-run --json` and audit the exact publish payload too.
- If tracked or npm-published files contain local/private-style literals, stop and clean them before staging or pushing.
- Prefer neutral placeholders such as `example.invalid`, `example.test`, and `EXAMPLE_*` in docs, tests, fixtures, and examples.
- Before any commit or push, show the planned command sequence and wait for user approval.