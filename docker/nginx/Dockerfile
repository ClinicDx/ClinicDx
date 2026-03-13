# =============================================================================
# ClinicDx V1 — nginx reverse proxy container
# =============================================================================
# Serves the OpenMRS frontend bundle and proxies /clinicdx-api → middleware
# and /openmrs → OpenMRS backend.  All logs are JSON on stdout/stderr.
# =============================================================================

FROM nginx:1.27-alpine

LABEL maintainer="ClinicDx" \
      version="1.0.0" \
      description="ClinicDx nginx reverse proxy"

# Remove default nginx configuration
RUN rm -f /etc/nginx/conf.d/default.conf

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy nginx configuration
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf

# Copy frontend static bundle (built separately, staged here)
# In docker-compose.yml this is populated from services/frontend/
COPY services/frontend/ /usr/share/nginx/html/

# SSL certificates directory (mounted as volume in docker-compose)
RUN mkdir -p /etc/nginx/certs

EXPOSE 80 443

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sfk https://localhost/openmrs/ || curl -sf http://localhost/ || exit 1
