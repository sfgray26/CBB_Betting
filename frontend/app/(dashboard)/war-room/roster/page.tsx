'use client'

import { Users } from 'lucide-react'

export default function RosterPage() {
  return (
    <div className="min-h-screen bg-black">
      <div className="max-w-2xl mx-auto p-6">
        <div className="flex items-center gap-2 mb-6">
          <Users className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">My Roster</span>
        </div>
        <div className="bg-[#202020] p-8 text-center">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#494949]">Coming Next</p>
          <p className="text-[#7D7D7D] text-sm mt-2">Slot-by-slot roster view</p>
        </div>
      </div>
    </div>
  )
}
