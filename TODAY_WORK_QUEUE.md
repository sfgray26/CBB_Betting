
**10:15 AM** - BUILD FIX: anyio==4.3.0 -> anyio>=4.5.0 (5e11538). fastapi-mcp depends on mcp which requires anyio>=4.5.0. Second dependency conflict unblocked today. Pushed to stable/cbb-prod.
**10:20 AM** - Updated deploy workflow to trigger on stable/cbb-prod (2b06e53). Merged stable/cbb-prod -> main to register workflow on default branch.
**10:25 AM** - GitHub Actions deploy triggered and running! Build job in progress.
