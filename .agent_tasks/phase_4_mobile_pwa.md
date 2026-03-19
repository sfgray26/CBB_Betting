# Task: Phase 4 — Mobile Responsiveness & PWA

**STATUS: OPEN**
**Assigned to:** Claude Code (Master Architect)
**Estimated:** 1-2 sessions
**Priority:** HIGH — tournament active, mobile users need dashboard access

## Goal

Make the Next.js dashboard fully usable on mobile (phones/tablets) and installable as a PWA.

## Dependencies

- **Kimi CLI** must produce `frontend/PHASE4_SPEC.md` first (delegated in HANDOFF EMAC-072)
- Read `frontend/PHASE4_SPEC.md` before starting any implementation

## Files to Modify

```
frontend/app/layout.tsx              # viewport meta tag, PWA link tags
frontend/components/layout/sidebar.tsx  # mobile drawer / hamburger
frontend/components/layout/header.tsx   # hamburger button (mobile)
frontend/components/layout/layout.tsx   # conditional sidebar display
frontend/public/manifest.json        # CREATE — PWA manifest
frontend/public/icons/               # CREATE — app icons (192, 512px)
```

## Success Criteria

- [ ] `<meta name="viewport">` present in root layout
- [ ] Sidebar collapses to drawer on screens < 768px (hamburger toggle)
- [ ] All buttons/links have min-h-[44px] touch target
- [ ] DataTable wrapped in `overflow-x-auto` for horizontal scroll on mobile
- [ ] `manifest.json` exists at `/public/manifest.json`
- [ ] `./scripts/validate_frontend.sh` — all PASS, no new BLOCKs
- [ ] TypeScript compiles: `cd frontend && npx tsc --noEmit`

## Test Command

```bash
# From repo root:
./scripts/validate_frontend.sh
cd frontend && npx tsc --noEmit
```

## Notes

- Sidebar: use `useState` to toggle open/closed. On mobile, show overlay backdrop.
- Design system: use existing zinc/amber palette — no new colors.
- PWA icons: amber-on-zinc ("CBB EDGE" text or monogram).
- Do NOT change any backend files.
