# ============================================================================
# CBB Edge Analyzer - Multi-service Docker Image
# Used by API, Worker, and Scheduler services on Railway
# Build context: repo root (files are in cbb-edge-analyzer/ subdirectory)
# ============================================================================

# -----------------------------------------------------------------------------
# Base Stage - Node setup
# -----------------------------------------------------------------------------
FROM node:20-alpine AS base
RUN apk update && apk add --no-cache openssl
RUN npm install -g pnpm@9.0.0

# -----------------------------------------------------------------------------
# Builder Stage - Install deps and build everything
# -----------------------------------------------------------------------------
FROM base AS builder
WORKDIR /app

# Cache bust - forces rebuild when this value changes
ARG CACHE_BUST=5

# Copy all source files from the subdirectory
COPY cbb-edge-analyzer/ .

# Force clean install (no cache)
RUN rm -rf node_modules packages/*/node_modules apps/*/node_modules

# Install all dependencies (dev + prod) - fresh install
RUN pnpm install --no-frozen-lockfile

# Generate Prisma client
RUN cd packages/infrastructure && npx prisma generate

# Build packages (in dependency order)
RUN pnpm --filter @cbb/core build
RUN pnpm --filter @cbb/infrastructure build

# Build apps (worker before api because api depends on worker)
RUN pnpm --filter @cbb/worker build
RUN pnpm --filter @cbb/api build

# -----------------------------------------------------------------------------
# Production Stage - Minimal runtime image
# -----------------------------------------------------------------------------
FROM node:20-alpine AS production
RUN apk update && apk add --no-cache openssl
RUN npm install -g pnpm@9.0.0

WORKDIR /app

# Copy package files for production install
COPY cbb-edge-analyzer/package.json cbb-edge-analyzer/pnpm-workspace.yaml cbb-edge-analyzer/turbo.json ./
COPY cbb-edge-analyzer/packages/core/package.json ./packages/core/
COPY cbb-edge-analyzer/packages/infrastructure/package.json ./packages/infrastructure/
COPY cbb-edge-analyzer/apps/api/package.json ./apps/api/
COPY cbb-edge-analyzer/apps/worker/package.json ./apps/worker/

# Copy built artifacts
COPY --from=builder /app/packages/core/dist ./packages/core/dist
COPY --from=builder /app/packages/infrastructure/dist ./packages/infrastructure/dist
COPY --from=builder /app/packages/infrastructure/prisma ./packages/infrastructure/prisma
COPY --from=builder /app/apps/api/dist ./apps/api/dist
COPY --from=builder /app/apps/worker/dist ./apps/worker/dist

# Install production dependencies only - fresh install
RUN rm -rf node_modules && pnpm install --prod --no-frozen-lockfile

# Generate Prisma client for production
RUN cd packages/infrastructure && npx prisma@5.22.0 generate

# Set environment
ENV NODE_ENV=production
ENV PORT=3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => r.statusCode === 200 ? process.exit(0) : process.exit(1))"

# Default command (overridden by Railway service config)
# Run db push first to create tables from schema, then start the server
CMD ["sh", "-c", "cd packages/infrastructure && npx prisma@5.22.0 db push --accept-data-loss && cd /app && node apps/api/dist/server.js"]
