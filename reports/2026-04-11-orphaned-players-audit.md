# Orphaned Players Audit (April 11, 2026)

## Overview
- **Total Orphans**: 366
- **Source**: `position_eligibility` records with `bdl_player_id IS NULL`.

## Key Categories Identified
1. **Multi-Role/Two-Way Players**:
   - Michael Lorenzen (Batter/Pitcher) - `469.p.1000005`, `469.p.9949`
   - Shohei Ohtani (Batter/Pitcher) - `469.p.1000001`, `469.p.1000002`
   *These likely fail to link because BDL uses a single ID while Yahoo splits them.*

2. **International Prospects & Signings**:
   - Adolfo Sanchez, Adriel Radney, Chen-Wei Lin, Ching-Hsien Ko, Hyungchan Um, etc.
   *Many of these are likely too new or too deep in the minors for the current BDL player mapping.*

3. **Retired/Edge Case Players**:
   - A significant portion of the names appear to be lower-tier prospects or retired players still present in the Yahoo ecosystem.

## Recommendation
- **Shohei Ohtani/Michael Lorenzen**: These require a manual override or a specific mapping logic to handle the Yahoo "Batter/Pitcher" split.
- **Prospects**: If these players aren't active in the MLB, the "0 linked" status from the fuzzy linker is technically correct as there is no corresponding BDL ID to link to.

## Full List (Sample)
- Aaron Watson (469.p.65763)
- Adolfo Sanchez (469.p.65017)
- Adriel Radney (469.p.64919)
- ... (See tool output for full list of 366)
