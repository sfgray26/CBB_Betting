import { SimulateResult } from "../../lib/types";

interface TeamStats {
  name: string;
  wins: number;
  losses: number;
  ties: number;
  logo?: string;
}

interface CategoryBattlefieldProps {
  myTeam: TeamStats | null;
  opponent: TeamStats | null;
  simulationResult?: SimulateResult | null;
}

export function CategoryBattlefield({ myTeam, opponent, simulationResult }: CategoryBattlefieldProps) {
  if (!myTeam || !opponent) {
    return (
      <div className="bg-[#1A1A1A] border border-[#333333] rounded-lg p-6">
        <p className="text-[#7D7D7D]">Loading category battlefield...</p>
      </div>
    );
  }

  return (
    <div className="bg-[#1A1A1A] border border-[#333333] rounded-lg p-6">
      <h2 className="text-[#FFC000] font-bold text-xl mb-4">Category Battlefield</h2>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <div className="text-[#7D7D7D] text-sm uppercase tracking-wider">{myTeam.name}</div>
          <div className="text-2xl font-bold text-white">{myTeam.wins}-{myTeam.losses}</div>
        </div>
        <div className="text-center flex items-center justify-center">
          <div className="text-[#FFC000] text-lg font-bold">VS</div>
        </div>
        <div className="text-center">
          <div className="text-[#7D7D7D] text-sm uppercase tracking-wider">{opponent.name}</div>
          <div className="text-2xl font-bold text-white">{opponent.wins}-{opponent.losses}</div>
        </div>
      </div>

      {simulationResult && (
        <div className="mt-4 p-4 bg-[#0D0D0D] rounded-lg">
          <div className="text-[#FFC000] font-bold mb-2">Simulation Breakdown</div>
          <div className="text-sm text-[#7D7D7D]">
            Win probability: {(simulationResult.win_probability * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-[#7D7D7D] mt-1">
            Based on {simulationResult.n_sims.toLocaleString()} simulations
          </div>
        </div>
      )}
    </div>
  );
}
