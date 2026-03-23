# Task: Phase 5 — Frontend Polish & Streamlit Decommission

**STATUS: OPEN** (after Phase 4 complete)
**Assigned to:** Claude Code (Master Architect)
**Estimated:** 2-3 sessions
**Priority:** MEDIUM (post-tournament)

## Goal

Production-harden the Next.js frontend and retire Streamlit once parity is confirmed.

## Sub-tasks

### 5.1 Error Boundaries
- Add React `ErrorBoundary` component wrapping each page's main content
- Catch unhandled exceptions → show user-friendly "Something went wrong" card
- File: `frontend/components/ui/error-boundary.tsx` (create)
- Wire in each `(dashboard)/*/page.tsx`

### 5.2 Suspense Fallbacks
- Add `<Suspense fallback={<PageSkeleton />}>` around async components
- Create `frontend/components/ui/page-skeleton.tsx`

### 5.3 TypeScript Strictness
- Run `cd frontend && npx tsc --noEmit --strict`
- Fix any strict-mode violations
- Enable `"strict": true` in `frontend/tsconfig.json`

### 5.4 Streamlit Decommission Checklist
Confirm parity before decommissioning Streamlit (`dashboard/` directory):
- [ ] All 13 Streamlit pages have Next.js equivalents
- [ ] Export all Streamlit-only data (performance snapshots, CLV history)
- [ ] Update Railway to remove Streamlit deployment
- [ ] Update `README.md` and `INSTALL.md`

## Success Criteria

- [ ] `./scripts/validate_frontend.sh` — all PASS, 0 warnings
- [ ] `npx tsc --noEmit` — 0 errors
- [ ] Each page renders correctly at 320px width (mobile test)
- [ ] Lighthouse performance score ≥ 80

## Notes

- Do NOT rush Streamlit decommission — Streamlit stays live until Next.js parity confirmed
- Keep `dashboard/` in repo even after decommission (archive, not delete)
