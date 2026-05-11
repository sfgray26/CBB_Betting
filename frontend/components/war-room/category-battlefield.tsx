import { MatchupResponse, MatchupSimulateResponse } from "@/lib/types";

interface CategoryBattlefieldProps {
  data: MatchupResponse;
  simulate?: MatchupSimulateResponse | null;
}

export function CategoryBattlefield({ data, simulate }: CategoryBattlefieldProps) {
  if (!data?.my_team || !data?.opponent) {
    return (
      <div className="bg-[#1A1A1A] border border-[#333333] rounded-lg p-6">
        <p className="text-[#7D7D7D] text-base">Loading category battlefield...</p>
      </div>
    );
  }

  const myTeam = data.my_team;
  const opponent = data.opponent;

  return (
    <div className="bg-[#1A1A1A] border border-[#333333] rounded-lg p-6">
      <h2 className="text-[#FFC000] font-bold text-2xl mb-4">Category Battlefield</h2>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <div className="text-[#7D7D7D] text-base uppercase tracking-wider">{myTeam.team_name}</div>
          <div className="text-2xl font-bold text-white">{myTeam.stats?.W ?? 0}-{myTeam.stats?.L ?? 0}</div>
        </div>
        <div className="text-center flex items-center justify-center">
          <div className="text-[#FFC000] text-lg font-bold">VS</div>
        </div>
        <div className="text-center">
          <div className="text-[#7D7D7D] text-base uppercase tracking-wider">{opponent.team_name}</div>
          <div className="text-2xl font-bold text-white">{opponent.stats?.W ?? 0}-{opponent.stats?.L ?? 0}</div>
        </div>
      </div>

      {simulate && (
        <div className="mt-4 p-4 bg-[#0D0D0D] rounded-lg">
          <div className="text-[#FFC000] font-bold mb-2 text-base">Simulation Breakdown</div>
          <div className="text-base text-[#7D7D7D]">
            Win probability: {(simulate.win_prob * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-[#7D7D7D] mt-1">
            Based on Monte Carlo simulation
          </div>
        </div>
      )}
    </div>
  );
}
