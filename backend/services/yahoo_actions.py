"""Validated Yahoo roster mutations using the canonical Yahoo client."""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

from backend.fantasy_baseball.yahoo_client_resilient import (
    YahooAPIError,
    YahooFantasyClient,
    get_yahoo_client,
)

logger = logging.getLogger(__name__)

ActionType = Literal["ADD", "DROP", "ADD_DROP"]

ACTIVE_LINEUP_SLOTS = {
    "C", "1B", "2B", "3B", "SS", "CI", "MI", "OF", "UTIL", "SP", "RP", "P"
}
BENCH_IL_SLOTS = {"BN", "IL", "IL60", "IL10", "NA", "DL"}

ERROR_CODES = {
    "INVALID_REQUEST": "invalid_request",
    "ROSTER_FULL": "roster_full",
    "PLAYER_NOT_AVAILABLE": "player_not_available",
    "PLAYER_NOT_ON_ROSTER": "player_not_on_roster",
    "PLAYER_UNDROPPABLE": "player_undroppable",
    "POSITION_INELIGIBLE": "position_ineligible",
    "YAHOO_API_ERROR": "yahoo_api_error",
}

WARNING_CODES = {
    "POSITION_DEFAULTED_TO_BN": "position_defaulted_to_bn",
    "PLAYER_ON_WAIVERS": "player_on_waivers",
    "DROP_PLAYER_INACTIVE": "drop_player_inactive",
}


@dataclass
class ActionError:
    code: str
    message: str


@dataclass
class ActionWarning:
    code: str
    message: str


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ActionError] = field(default_factory=list)
    warnings: List[ActionWarning] = field(default_factory=list)
    add_player_available: bool = False
    add_player_on_waivers: bool = False
    drop_player_on_roster: bool = False
    position_eligible: bool = False
    roster_space_available: bool = False


@dataclass
class ActionResult:
    success: bool
    transaction_id: Optional[str] = None
    roster_state: Dict[str, Any] = field(default_factory=dict)
    errors: List[ActionError] = field(default_factory=list)
    warnings: List[ActionWarning] = field(default_factory=list)
    rollback_attempted: bool = False
    manual_action_required: bool = False
    add_succeeded: bool = False
    drop_succeeded: bool = False
    rollback_succeeded: bool = False
    execution_time_et: Optional[str] = None


def _numeric_player_id(player_key: str) -> str:
    """Return the exact numeric suffix from a validated Yahoo player key."""
    return player_key.rsplit(".p.", 1)[-1]


class YahooActionsService:
    """Two-phase roster action service: validate, then submit one mutation."""

    def __init__(self) -> None:
        self._client: Optional[YahooFantasyClient] = None
        self._team_key: Optional[str] = None

    def _get_client(self) -> YahooFantasyClient:
        if self._client is None:
            self._client = get_yahoo_client()
        return self._client

    def _get_team_key(self) -> str:
        if self._team_key is None:
            self._team_key = os.getenv("YAHOO_TEAM_KEY") or self._get_client().get_my_team_key()
        return self._team_key or ""

    @staticmethod
    def _find_roster_player(roster: List[dict], player_key: str) -> Optional[dict]:
        numeric_id = _numeric_player_id(player_key)
        return next(
            (
                player
                for player in roster
                if player.get("player_key") == player_key
                or str(player.get("player_id") or "") == numeric_id
            ),
            None,
        )

    async def validate_roster_action(
        self,
        action: ActionType,
        add_player_id: Optional[str],
        drop_player_id: Optional[str],
        position: Optional[str],
    ) -> ValidationResult:
        """Phase 1: perform read-only validation with no Yahoo mutation."""
        result = ValidationResult(is_valid=True)

        if action in ("ADD", "ADD_DROP") and not add_player_id:
            result.errors.append(ActionError(
                ERROR_CODES["INVALID_REQUEST"],
                f"add_player_id is required for {action}",
            ))
        if action in ("DROP", "ADD_DROP") and not drop_player_id:
            result.errors.append(ActionError(
                ERROR_CODES["INVALID_REQUEST"],
                f"drop_player_id is required for {action}",
            ))
        if add_player_id and drop_player_id and add_player_id == drop_player_id:
            result.errors.append(ActionError(
                ERROR_CODES["INVALID_REQUEST"],
                "add_player_id and drop_player_id must be different",
            ))
        if result.errors:
            result.is_valid = False
            return result

        client = self._get_client()
        team_key = self._get_team_key()
        if not team_key:
            return ValidationResult(
                is_valid=False,
                errors=[ActionError(
                    ERROR_CODES["YAHOO_API_ERROR"],
                    "Yahoo team key is not configured",
                )],
            )

        try:
            roster = await asyncio.to_thread(client.get_roster, team_key)
        except YahooAPIError as exc:
            return ValidationResult(
                is_valid=False,
                errors=[ActionError(
                    ERROR_CODES["YAHOO_API_ERROR"],
                    f"Failed to fetch roster: {exc}",
                )],
            )

        current_roster_size = len(roster)
        if action == "ADD":
            result.roster_space_available = current_roster_size < self._get_max_roster_size()
            if not result.roster_space_available:
                result.errors.append(ActionError(
                    ERROR_CODES["ROSTER_FULL"],
                    f"Roster is full ({current_roster_size}/{self._get_max_roster_size()}). "
                    "Use ADD_DROP to preserve roster size.",
                ))
        else:
            result.roster_space_available = True

        add_player = None
        if add_player_id:
            try:
                add_player = await asyncio.to_thread(client.get_player, add_player_id)
            except YahooAPIError as exc:
                result.errors.append(ActionError(
                    ERROR_CODES["PLAYER_NOT_AVAILABLE"],
                    f"Player {add_player_id} could not be loaded: {exc}",
                ))
            else:
                status = str(add_player.get("status") or "").upper()
                if status == "T":
                    result.errors.append(ActionError(
                        ERROR_CODES["PLAYER_NOT_AVAILABLE"],
                        f"Player {add_player_id} is already rostered",
                    ))
                else:
                    result.add_player_available = True
                    result.add_player_on_waivers = status == "W"
                    if result.add_player_on_waivers:
                        result.warnings.append(ActionWarning(
                            WARNING_CODES["PLAYER_ON_WAIVERS"],
                            f"Player {add_player.get('name', add_player_id)} is on waivers; "
                            "Yahoo may create a claim instead of an immediate add.",
                        ))

        if drop_player_id:
            drop_player = self._find_roster_player(roster, drop_player_id)
            result.drop_player_on_roster = drop_player is not None
            if drop_player is None:
                result.errors.append(ActionError(
                    ERROR_CODES["PLAYER_NOT_ON_ROSTER"],
                    f"Player {drop_player_id} is not on your roster.",
                ))
            elif drop_player.get("is_undroppable"):
                result.errors.append(ActionError(
                    ERROR_CODES["PLAYER_UNDROPPABLE"],
                    f"Player {drop_player.get('name', drop_player_id)} is on Yahoo's "
                    "can't-cut list.",
                ))
            elif drop_player.get("selected_position") in BENCH_IL_SLOTS - {"BN"}:
                selected_position = drop_player["selected_position"]
                result.warnings.append(ActionWarning(
                    WARNING_CODES["DROP_PLAYER_INACTIVE"],
                    f"Player {drop_player.get('name', drop_player_id)} is on "
                    f"{selected_position}. Dropping may affect roster rules.",
                ))

        target_position = position or "BN"
        if position is None and action in ("ADD", "ADD_DROP"):
            result.warnings.append(ActionWarning(
                WARNING_CODES["POSITION_DEFAULTED_TO_BN"],
                "No position specified; Yahoo adds the player to the bench.",
            ))

        if add_player and target_position in ACTIVE_LINEUP_SLOTS:
            eligible_positions = [
                item for item in add_player.get("positions", []) if isinstance(item, str)
            ]
            result.position_eligible = (
                target_position in eligible_positions
                or self._check_utility_eligibility(target_position, eligible_positions)
            )
            if not result.position_eligible:
                result.errors.append(ActionError(
                    ERROR_CODES["POSITION_INELIGIBLE"],
                    f"Player {add_player.get('name', add_player_id)} is not eligible for "
                    f"{target_position}. Eligible positions: "
                    f"{', '.join(eligible_positions) or 'None'}",
                ))
        else:
            result.position_eligible = True

        result.is_valid = not result.errors
        return result

    @staticmethod
    def _get_max_roster_size() -> int:
        """Conservative league-size fallback used only for standalone ADD."""
        return 23

    @staticmethod
    def _check_utility_eligibility(position: str, eligible: List[str]) -> bool:
        if position != "UTIL":
            return False
        return bool({"1B", "2B", "3B", "SS", "CI", "MI", "OF"} & set(eligible))

    async def execute_action(
        self,
        action: ActionType,
        add_player_id: Optional[str],
        drop_player_id: Optional[str],
        position: Optional[str],
    ) -> ActionResult:
        """Phase 2: submit exactly one client-compatible Yahoo transaction."""
        execution_time = datetime.now(ZoneInfo("America/New_York")).isoformat()
        validation = await self.validate_roster_action(
            action=action,
            add_player_id=add_player_id,
            drop_player_id=drop_player_id,
            position=position,
        )
        result = ActionResult(
            success=False,
            warnings=validation.warnings,
            execution_time_et=execution_time,
        )
        if not validation.is_valid:
            result.errors = validation.errors
            return result

        client = self._get_client()
        team_key = self._get_team_key()

        try:
            if action == "ADD":
                success = await asyncio.to_thread(
                    client.add_drop_player,
                    add_player_key=add_player_id,
                    drop_player_key=None,
                    team_key=team_key,
                )
                result.add_succeeded = bool(success)
            elif action == "DROP":
                success = await asyncio.to_thread(
                    client.drop_player,
                    player_key=drop_player_id,
                    team_key=team_key,
                )
                result.drop_succeeded = bool(success)
            else:
                success = await asyncio.to_thread(
                    client.add_drop_player,
                    add_player_key=add_player_id,
                    drop_player_key=drop_player_id,
                    team_key=team_key,
                )
                result.add_succeeded = bool(success)
                result.drop_succeeded = bool(success)
        except YahooAPIError as exc:
            result.errors.append(ActionError(
                ERROR_CODES["YAHOO_API_ERROR"],
                f"Yahoo {action.lower()} failed: {exc}",
            ))
            result.manual_action_required = True
            return result

        if not success:
            result.errors.append(ActionError(
                ERROR_CODES["YAHOO_API_ERROR"],
                f"Yahoo {action.lower()} returned an unsuccessful result",
            ))
            result.manual_action_required = True
            return result

        result.success = True
        result.transaction_id = self._generate_transaction_id(
            add_player_id,
            drop_player_id,
        )
        try:
            updated_roster = await asyncio.to_thread(client.get_roster, team_key)
            result.roster_state = {
                "player_count": len(updated_roster),
                "players": [
                    {
                        "player_key": player.get("player_key"),
                        "player_id": player.get("player_id"),
                        "name": player.get("name"),
                        "selected_position": player.get("selected_position"),
                        "status": player.get("status"),
                    }
                    for player in updated_roster
                ],
            }
        except YahooAPIError as exc:
            logger.warning("Roster action succeeded but refresh failed: %s", exc)
            result.warnings.append(ActionWarning(
                ERROR_CODES["YAHOO_API_ERROR"],
                "Yahoo accepted the action, but the refreshed roster could not be loaded.",
            ))
        return result

    @staticmethod
    def _generate_transaction_id(
        add_id: Optional[str],
        drop_id: Optional[str],
    ) -> str:
        timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d%H%M%S%f")
        parts = ["txn", timestamp]
        if add_id:
            parts.append(f"add_{_numeric_player_id(add_id)}")
        if drop_id:
            parts.append(f"drop_{_numeric_player_id(drop_id)}")
        return "-".join(parts)


_service_instance: Optional[YahooActionsService] = None


def get_yahoo_actions_service() -> YahooActionsService:
    global _service_instance
    if _service_instance is None:
        _service_instance = YahooActionsService()
    return _service_instance
