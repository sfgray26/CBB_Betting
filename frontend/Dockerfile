# =============================================================================
# CBB Edge — Next.js Frontend Dockerfile
# Multi-stage build for Railway deployment.
#
# Requires next.config.ts: output: 'standalone'
#
# Environment variables (set in Railway service):
#   NEXT_PUBLIC_API_URL  — Railway backend URL, e.g. https://cbbbetting-production.up.railway.app
# =============================================================================

FROM node:20-alpine AS base

# ── Stage 1: Install dependencies ───────────────────────────────────────────
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# ── Stage 2: Build ───────────────────────────────────────────────────────────
FROM base AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* vars are inlined at build time — must be provided here.
# Railway injects them as build args automatically when set as service variables.
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN npm run build

# ── Stage 3: Production runner ───────────────────────────────────────────────
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

RUN addgroup --system --gid 1001 nodejs
RUN adduser  --system --uid 1001 nextjs

# Copy standalone output (minimal, no node_modules copy needed)
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
