# ==========================================
# Stage 1: Builder
# ==========================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY package*.json bun.lock* ./

# Install dependencies for compilation
RUN npm ci || npm install

# Copy application sources
COPY . .

# Build client bundle & server executable
RUN npm run build

# ==========================================
# Stage 2: Production Runner
# ==========================================
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

# Install production dependencies only
COPY package*.json ./
RUN npm ci --omit=dev || npm install --omit=dev

# Copy compiled bundles and configs from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/metadata.json ./metadata.json
COPY --from=builder /app/firebase-applet-config.json ./firebase-applet-config.json
COPY --from=builder /app/firebase-applet-config.example.json ./firebase-applet-config.example.json

# Create persistent data directory with permissions for node user
RUN mkdir -p /app/data && chown -R node:node /app/data

# Persistent storage mount point for database
VOLUME ["/app/data"]

# Run as non-root user for security
USER node

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT}/api/health || exit 1

# Start production server
CMD ["node", "dist/server.cjs"]
