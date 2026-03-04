# CLAUDE.md

This file provides guidance for AI assistants (Claude and others) working in this repository.

## Repository Status

This is a freshly initialized repository (`scruffyjay/claude`). It currently contains no source code. This CLAUDE.md will be updated as the project evolves.

---

## Git Workflow

### Branch Naming Convention

Feature branches follow the pattern:

```
claude/<description>-<session-id>
```

Example: `claude/claude-md-mmc5o5l8j9zbwysw-yYgvv`

### Standard Git Operations

**Pushing changes:**
```bash
git push -u origin <branch-name>
```

**Fetching a specific branch:**
```bash
git fetch origin <branch-name>
```

**Commit message format:**
- Use the imperative mood: "Add feature" not "Added feature"
- Keep the subject line under 72 characters
- Separate subject from body with a blank line when more detail is needed

### Push Rules

- Never push directly to `main` or `master` without explicit permission
- All work should be developed on feature branches
- Branch names must start with `claude/` when working in AI-assisted sessions

---

## Development Guidelines

### General Conventions

- Prefer editing existing files over creating new ones
- Avoid over-engineering — implement only what is needed for the current task
- Do not add speculative features, extra comments, or docstrings to unchanged code
- Keep solutions minimal and focused

### Code Quality

- Write secure code: avoid command injection, XSS, SQL injection, and other OWASP top 10 vulnerabilities
- Validate input at system boundaries (user input, external APIs); trust internal guarantees
- Avoid adding error handling for scenarios that cannot happen

### File Management

- Do not create unnecessary files
- Remove unused code rather than commenting it out or adding backwards-compatibility shims
- Avoid creating documentation files (README, .md) unless explicitly requested

---

## AI Assistant Instructions

When working in this repository:

1. **Read before editing** — always read a file before modifying it
2. **Understand before suggesting** — understand existing code before proposing changes
3. **Minimal changes** — make only the changes needed; avoid refactoring unrelated code
4. **Confirm risky actions** — before destructive or irreversible operations (force-push, branch deletion, dropping data), confirm with the user
5. **Parallel tool use** — when multiple independent operations are needed, run them in parallel

### Risky Actions Requiring Confirmation

- `git push --force`
- `git reset --hard`
- Deleting branches or files
- Modifying CI/CD pipelines
- Any action visible to other collaborators (PRs, comments, messages)

---

## Notes

- Remote origin: `http://local_proxy@127.0.0.1:42253/git/scruffyjay/claude`
- Update this file as the project grows to reflect actual tech stack, commands, and conventions
