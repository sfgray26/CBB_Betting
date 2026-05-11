#!/usr/bin/env python3
"""
Backfill missing player_identities rows for canonical projection board misses.

This script targets the exact normalized-name misses surfaced by
scripts/validate_canonical_projections.py.

Resolution strategy:
1. Prefer exact player_id_mapping matches (by yahoo_id when available, otherwise by
   normalized_name).
2. Insert a new player_identities row using the strongest non-conflicting identifiers.
   If an MLBAM / Yahoo GUID / FanGraphs identifier is already occupied by a different
   identity row, suppress that identifier rather than skipping the player entirely.
3. For remaining unresolved board names (alias / punctuation / suffix / new-player
   misses), insert a placeholder identity row with a deterministic synthetic negative
   yahoo_id and the board name as full_name / normalized_name.

Source board:
- If a production player_board table exists, use it.
- Otherwise fall back to backend.fantasy_baseball.player_board.get_board(), which is
  the same source used by validate_canonical_projections.py in this repo.

Idempotent: safe to run multiple times.
Use --dry-run to execute inside a rollback-only transaction.
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.fantasy_baseball.player_board import get_board


@dataclass(frozen=True)
class BoardPlayer:
    full_name: str
    normalized_name: str
    yahoo_id: str | None
    team: str | None
    player_type: str | None
    source: str


@dataclass(frozen=True)
class MappingRow:
    yahoo_id: str | None
    yahoo_guid: str | None
    full_name: str
    normalized_name: str
    mlbam_id: int | None
    fangraphs_id: str | None


@dataclass(frozen=True)
class IdentityRow:
    id: int
    yahoo_id: str | None
    yahoo_guid: str | None
    mlbam_id: int | None
    fangraphs_id: str | None
    full_name: str
    normalized_name: str
    active: bool


@dataclass(frozen=True)
class InsertPlan:
    player_name: str
    normalized_name: str
    yahoo_id: str
    yahoo_guid: str | None
    mlbam_id: int | None
    fangraphs_id: str | None
    team: str | None
    player_type: str | None
    source: str
    strategy: str
    notes: tuple[str, ...]


def _require_database_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    return db_url


def _normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _table_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = :table_name
            ORDER BY ordinal_position
            """
        ),
        {"table_name": table_name},
    ).fetchall()
    return {row[0] for row in rows}


def _choose_mapping(existing: MappingRow | None, candidate: MappingRow) -> MappingRow:
    if existing is None:
        return candidate

    def score(row: MappingRow) -> tuple[int, int, int, int]:
        return (
            1 if row.yahoo_id else 0,
            1 if row.yahoo_guid else 0,
            1 if row.mlbam_id and row.mlbam_id > 0 else 0,
            1 if row.fangraphs_id else 0,
        )

    return candidate if score(candidate) > score(existing) else existing


def _load_board_players(conn) -> list[BoardPlayer]:
    columns = _table_columns(conn, "player_board")
    players: list[BoardPlayer] = []

    if columns:
        has_player_type = "player_type" in columns
        where_type = ""
        if has_player_type:
            where_type = (
                "WHERE pb.player_type IN ('SP','RP','P','C','1B','2B','3B','SS','OF','DH','Util')"
            )
        query = f"""
            SELECT DISTINCT
                pb.player_id::text AS yahoo_id,
                pb.player_name,
                pb.team,
                {('pb.player_type' if has_player_type else 'NULL AS player_type')}
            FROM player_board pb
            {where_type}
            ORDER BY pb.player_name
        """
        rows = conn.execute(text(query)).fetchall()
        for yahoo_id, player_name, team, player_type in rows:
            if not player_name:
                continue
            players.append(
                BoardPlayer(
                    full_name=player_name,
                    normalized_name=_normalize_name(player_name),
                    yahoo_id=str(yahoo_id) if yahoo_id is not None else None,
                    team=team,
                    player_type=player_type,
                    source="player_board",
                )
            )
        return players

    for player in get_board(apply_park_factors=False):
        player_name = player.get("name")
        if not player_name:
            continue
        player_type = player.get("type")
        if player_type not in {"batter", "pitcher"}:
            continue
        players.append(
            BoardPlayer(
                full_name=player_name,
                normalized_name=_normalize_name(player_name),
                yahoo_id=None,
                team=player.get("team"),
                player_type=player_type,
                source="get_board",
            )
        )
    return players


def _load_mapping_rows(conn) -> tuple[dict[str, MappingRow], dict[str, MappingRow], dict[str, str | None]]:
    columns = _table_columns(conn, "player_id_mapping")
    if not columns:
        return {}, {}, {}

    if "full_name" in columns:
        name_column = "full_name"
    elif "player_name" in columns:
        name_column = "player_name"
    else:
        raise RuntimeError("player_id_mapping is missing both full_name and player_name")

    yahoo_guid_column = "yahoo_key" if "yahoo_key" in columns else None
    fangraphs_column = "fangraphs_id" if "fangraphs_id" in columns else None
    normalized_column = "normalized_name" if "normalized_name" in columns else None

    query = f"""
        SELECT
            yahoo_id,
            {yahoo_guid_column or 'NULL'} AS yahoo_guid,
            {name_column} AS full_name,
            {normalized_column or 'NULL'} AS normalized_name,
            mlbam_id,
            {fangraphs_column or 'NULL'} AS fangraphs_id
        FROM player_id_mapping
        WHERE {name_column} IS NOT NULL
    """
    rows = conn.execute(text(query)).fetchall()

    by_yahoo_id: dict[str, MappingRow] = {}
    by_normalized_name: dict[str, MappingRow] = {}

    for row in rows:
        mapping = MappingRow(
            yahoo_id=str(row.yahoo_id) if row.yahoo_id is not None else None,
            yahoo_guid=row.yahoo_guid,
            full_name=row.full_name,
            normalized_name=_normalize_name(row.normalized_name or row.full_name),
            mlbam_id=row.mlbam_id,
            fangraphs_id=str(row.fangraphs_id) if row.fangraphs_id is not None else None,
        )
        if mapping.yahoo_id:
            by_yahoo_id[mapping.yahoo_id] = _choose_mapping(by_yahoo_id.get(mapping.yahoo_id), mapping)
        by_normalized_name[mapping.normalized_name] = _choose_mapping(
            by_normalized_name.get(mapping.normalized_name),
            mapping,
        )

    return by_yahoo_id, by_normalized_name, {
        "name_column": name_column,
        "yahoo_guid_column": yahoo_guid_column,
        "fangraphs_column": fangraphs_column,
        "normalized_column": normalized_column,
    }


def _load_identity_rows(conn) -> list[IdentityRow]:
    rows = conn.execute(
        text(
            """
            SELECT id, yahoo_id, yahoo_guid, mlbam_id, fangraphs_id, full_name, normalized_name, active
            FROM player_identities
            ORDER BY id
            """
        )
    ).fetchall()
    return [
        IdentityRow(
            id=row.id,
            yahoo_id=str(row.yahoo_id) if row.yahoo_id is not None else None,
            yahoo_guid=row.yahoo_guid,
            mlbam_id=row.mlbam_id,
            fangraphs_id=str(row.fangraphs_id) if row.fangraphs_id is not None else None,
            full_name=row.full_name,
            normalized_name=row.normalized_name,
            active=bool(row.active),
        )
        for row in rows
    ]


def _yahoo_constraint_name(conn) -> str:
    row = conn.execute(
        text(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = 'player_identities'::regclass
              AND contype = 'u'
              AND pg_get_constraintdef(oid) = 'UNIQUE (yahoo_id)'
            LIMIT 1
            """
        )
    ).fetchone()
    return row[0] if row else "player_identities_yahoo_id_key"


def _synthetic_yahoo_id(normalized_name: str, reserved_ids: set[str]) -> str:
    base = zlib.crc32(normalized_name.encode("utf-8"))
    suffix = 0
    while True:
        candidate = f"-9{base:010d}{suffix if suffix else ''}"
        if candidate not in reserved_ids:
            return candidate
        suffix += 1


def _resolve_identifier_conflict(
    identifier_name: str,
    identifier_value: Any,
    lookup: dict[Any, IdentityRow],
    target_normalized_name: str,
    notes: list[str],
) -> Any:
    if identifier_value in (None, ""):
        return None
    owner = lookup.get(identifier_value)
    if owner is None or owner.normalized_name == target_normalized_name:
        return identifier_value
    notes.append(
        f"suppressed {identifier_name}={identifier_value} (owned by {owner.full_name})"
    )
    return None


def _dedupe_board(players: list[BoardPlayer]) -> list[BoardPlayer]:
    deduped: dict[str, BoardPlayer] = {}
    for player in players:
        existing = deduped.get(player.normalized_name)
        if existing is None or (existing.yahoo_id is None and player.yahoo_id is not None):
            deduped[player.normalized_name] = player
    return list(deduped.values())


def _find_missing_players(players: list[BoardPlayer], identity_rows: list[IdentityRow]) -> list[BoardPlayer]:
    existing_names = {row.normalized_name for row in identity_rows if row.active}
    return [player for player in players if player.normalized_name not in existing_names]


def _build_plan(
    player: BoardPlayer,
    mapping_by_yahoo_id: dict[str, MappingRow],
    mapping_by_normalized_name: dict[str, MappingRow],
    identity_by_yahoo_id: dict[str, IdentityRow],
    identity_by_yahoo_guid: dict[str, IdentityRow],
    identity_by_mlbam: dict[int, IdentityRow],
    identity_by_fangraphs: dict[str, IdentityRow],
    reserved_yahoo_ids: set[str],
) -> InsertPlan:
    notes: list[str] = []
    mapping = None
    if player.yahoo_id:
        mapping = mapping_by_yahoo_id.get(player.yahoo_id)
    if mapping is None:
        mapping = mapping_by_normalized_name.get(player.normalized_name)

    strategy = "mapping_backfill" if mapping is not None else "placeholder_backfill"
    if mapping is not None:
        notes.append(f"mapping match: {mapping.full_name}")
    else:
        notes.append("no exact mapping match")

    preferred_yahoo_id = player.yahoo_id or (mapping.yahoo_id if mapping else None)
    if preferred_yahoo_id and preferred_yahoo_id in reserved_yahoo_ids:
        owner = identity_by_yahoo_id.get(preferred_yahoo_id)
        if owner is not None and owner.normalized_name != player.normalized_name:
            notes.append(f"suppressed yahoo_id={preferred_yahoo_id} (owned by {owner.full_name})")
            preferred_yahoo_id = None

    yahoo_id = preferred_yahoo_id or _synthetic_yahoo_id(player.normalized_name, reserved_yahoo_ids)
    if preferred_yahoo_id is None:
        notes.append(f"synthetic yahoo_id={yahoo_id}")

    yahoo_guid = _resolve_identifier_conflict(
        "yahoo_guid",
        mapping.yahoo_guid if mapping else None,
        identity_by_yahoo_guid,
        player.normalized_name,
        notes,
    )
    mlbam_id = _resolve_identifier_conflict(
        "mlbam_id",
        mapping.mlbam_id if mapping and mapping.mlbam_id and mapping.mlbam_id > 0 else None,
        identity_by_mlbam,
        player.normalized_name,
        notes,
    )
    fangraphs_id = _resolve_identifier_conflict(
        "fangraphs_id",
        mapping.fangraphs_id if mapping else None,
        identity_by_fangraphs,
        player.normalized_name,
        notes,
    )

    return InsertPlan(
        player_name=player.full_name,
        normalized_name=player.normalized_name,
        yahoo_id=yahoo_id,
        yahoo_guid=yahoo_guid,
        mlbam_id=mlbam_id,
        fangraphs_id=fangraphs_id,
        team=player.team,
        player_type=player.player_type,
        source=player.source,
        strategy=strategy,
        notes=tuple(notes),
    )


def run(dry_run: bool = False) -> int:
    engine = create_engine(_require_database_url(), pool_pre_ping=True)

    with engine.connect() as conn:
        board_players = _dedupe_board(_load_board_players(conn))
        mapping_by_yahoo_id, mapping_by_normalized_name, mapping_meta = _load_mapping_rows(conn)
        identity_rows = _load_identity_rows(conn)
        yahoo_constraint = _yahoo_constraint_name(conn)

    missing_players = _find_missing_players(board_players, identity_rows)

    identity_by_yahoo_id = {row.yahoo_id: row for row in identity_rows if row.yahoo_id}
    identity_by_yahoo_guid = {row.yahoo_guid: row for row in identity_rows if row.yahoo_guid}
    identity_by_mlbam = {row.mlbam_id: row for row in identity_rows if row.mlbam_id is not None}
    identity_by_fangraphs = {row.fangraphs_id: row for row in identity_rows if row.fangraphs_id}
    reserved_yahoo_ids = {row.yahoo_id for row in identity_rows if row.yahoo_id}

    plans: list[InsertPlan] = []
    for player in missing_players:
        plan = _build_plan(
            player,
            mapping_by_yahoo_id,
            mapping_by_normalized_name,
            identity_by_yahoo_id,
            identity_by_yahoo_guid,
            identity_by_mlbam,
            identity_by_fangraphs,
            reserved_yahoo_ids,
        )
        plans.append(plan)
        reserved_yahoo_ids.add(plan.yahoo_id)
        identity_by_yahoo_id[plan.yahoo_id] = IdentityRow(
            id=-1,
            yahoo_id=plan.yahoo_id,
            yahoo_guid=plan.yahoo_guid,
            mlbam_id=plan.mlbam_id,
            fangraphs_id=plan.fangraphs_id,
            full_name=plan.player_name,
            normalized_name=plan.normalized_name,
            active=True,
        )
        if plan.yahoo_guid:
            identity_by_yahoo_guid[plan.yahoo_guid] = identity_by_yahoo_id[plan.yahoo_id]
        if plan.mlbam_id is not None:
            identity_by_mlbam[plan.mlbam_id] = identity_by_yahoo_id[plan.yahoo_id]
        if plan.fangraphs_id:
            identity_by_fangraphs[plan.fangraphs_id] = identity_by_yahoo_id[plan.yahoo_id]

    print("=== Backfill missing player_identities ===")
    print(f"Board source rows:          {len(board_players)}")
    print(f"Missing identities:         {len(missing_players)}")
    print(f"Mapping rows available:     {len(mapping_by_normalized_name)}")
    print(f"Mapping columns:            {mapping_meta}")
    print(f"Yahoo conflict constraint:  {yahoo_constraint}")
    print()

    if plans:
        print("Planned inserts:")
        for plan in plans:
            print(
                f"- {plan.player_name} | strategy={plan.strategy} | "
                f"yahoo_id={plan.yahoo_id} | yahoo_guid={plan.yahoo_guid} | "
                f"mlbam_id={plan.mlbam_id} | fangraphs_id={plan.fangraphs_id}"
            )
            for note in plan.notes:
                print(f"    · {note}")
    else:
        print("No missing identities found. Nothing to do.")
        return 0

    insert_sql = text(
        f"""
        INSERT INTO player_identities (
            yahoo_guid,
            yahoo_id,
            mlbam_id,
            fangraphs_id,
            full_name,
            normalized_name,
            active,
            created_at,
            updated_at
        )
        VALUES (
            :yahoo_guid,
            :yahoo_id,
            :mlbam_id,
            :fangraphs_id,
            :full_name,
            :normalized_name,
            TRUE,
            NOW(),
            NOW()
        )
        ON CONFLICT ON CONSTRAINT {yahoo_constraint} DO NOTHING
        RETURNING id
        """
    )

    inserted = 0
    mapping_backfill = 0
    placeholder_backfill = 0

    with engine.connect() as conn:
        tx = conn.begin()
        try:
            for plan in plans:
                result = conn.execute(
                    insert_sql,
                    {
                        "yahoo_guid": plan.yahoo_guid,
                        "yahoo_id": plan.yahoo_id,
                        "mlbam_id": plan.mlbam_id,
                        "fangraphs_id": plan.fangraphs_id,
                        "full_name": plan.player_name,
                        "normalized_name": plan.normalized_name,
                    },
                ).fetchone()
                if result is not None:
                    inserted += 1
                    if plan.strategy == "mapping_backfill":
                        mapping_backfill += 1
                    else:
                        placeholder_backfill += 1

            if dry_run:
                tx.rollback()
            else:
                tx.commit()
        except Exception:
            tx.rollback()
            raise

    print()
    print("Summary:")
    print(f"  Planned rows:             {len(plans)}")
    print(f"  Executed inserts:         {inserted}")
    print(f"  Mapping-backed inserts:   {mapping_backfill}")
    print(f"  Placeholder inserts:      {placeholder_backfill}")
    print(f"  Expected misses after run: {max(len(missing_players) - inserted, 0)}")
    print("  Transaction mode:         DRY RUN (rolled back)" if dry_run else "  Transaction mode:         COMMIT")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill missing player_identities for canonical projection misses")
    parser.add_argument("--dry-run", action="store_true", help="Execute inserts inside a rollback-only transaction")
    args = parser.parse_args()
    raise SystemExit(run(dry_run=args.dry_run))
