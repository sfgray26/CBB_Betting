import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Required for Docker / Railway standalone deployment.
  // Produces a minimal self-contained build in .next/standalone/.
  output: 'standalone',
}

export default nextConfig
