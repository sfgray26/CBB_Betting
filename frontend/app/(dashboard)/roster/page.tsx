'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function RosterRedirectPage() {
  const router = useRouter()
  useEffect(() => {
    router.replace('/war-room/roster')
  }, [router])
  return null
}
