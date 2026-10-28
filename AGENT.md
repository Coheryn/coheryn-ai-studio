# Agent instructions

- Always use the latest stable, non-yanked releases of project packages and development tools; verify upstream versions, update manifests and lockfiles, adapt code as needed, and run the relevant checks before completing changes.
- Use `https://github.com/Coheryn/coheryn-ai-studio` as the canonical project repository URL and `https://coheryntech.com/products/coheryn-ai-studio` as the canonical project website URL in documentation and package metadata.
- Preserve third-party license/copyright notices and immutable reference bytes. Update `resources/licensing/` for copied assets or dependency changes, run `python3 tools/license_notices.py`, and include its complete legal payload in distributions. Required attribution must never be removed to satisfy a presentation scan.
