'use client'

import { ListFilter } from 'lucide-react'

export default function WaiverPage() {
  return (
    <div className="min-h-screen bg-black">
      <div className="max-w-4xl mx-auto p-6">
        <div className="flex items-center gap-2 mb-6">
          <ListFilter className="h-3.5 w-3.5 text-[#FFC000]" />
          <span className="text-xs font-bold tracking-widest uppercase text-[#FFC000]">Waiver Wire</span>
        </div>
        <div className="bg-[#202020] p-8 text-center">
          <p className="text-xs font-semibold tracking-widest uppercase text-[#494949]">Coming Next</p>
          <p className="text-[#7D7D7D] text-sm mt-2">Filter-driven free agent rankings by bubble category</p>
        </div>
      </div>
    </div>
  )
}
