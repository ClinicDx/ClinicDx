# =============================================================================
# ClinicDx V1 — Makefile
# =============================================================================
# Usage:
#   make up          Start full stack (GPU)
#   make up-cpu      Start full stack (CPU only)
#   make down        Stop all containers
#   make logs        Follow logs from all containers
#   make test        Run unit tests then integration tests
#   make test-unit   Run unit tests only (no Docker required)
#   make test-int    Run integration tests only (requires running stack)
#   make smoke       Run smoke test against the running stack
#   make backup      Snapshot the live /var/www/ClinicDx deployment
#   make prefetch    Pre-download all artifacts into Docker volumes
# =============================================================================

.PHONY: up up-cpu down restart logs ps \
        test test-unit test-int smoke \
        backup prefetch lint help

# ── Stack lifecycle ───────────────────────────────────────────────────────────

up:
	docker compose --profile gpu up -d
	@echo "Stack started (GPU). Run 'make logs' to follow output."

up-cpu:
	docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
	@echo "Stack started (CPU). Run 'make logs' to follow output."

down:
	docker compose down

restart:
	docker compose down
	docker compose --profile gpu up -d

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

# ── Testing ───────────────────────────────────────────────────────────────────

test: test-unit test-int
	@echo "All tests passed."

test-unit:
	@echo "--- Running unit tests ---"
	python3 -m pytest tests/unit/ -v --tb=short

test-int:
	@echo "--- Running integration tests ---"
	python3 -m pytest tests/integration/ -v --tb=short

smoke:
	@bash scripts/smoke_test.sh

# ── Maintenance ───────────────────────────────────────────────────────────────

backup:
	@bash scripts/backup.sh

prefetch:
	@bash scripts/download_artifacts.sh

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	python3 -m ruff check services/ tests/
	python3 -m mypy services/ --ignore-missing-imports

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "ClinicDx V1 — available targets:"
	@echo "  up          Start stack (GPU profile)"
	@echo "  up-cpu      Start stack (CPU profile, same Q8 model)"
	@echo "  down        Stop all containers"
	@echo "  restart     down + up"
	@echo "  logs        Follow all container logs"
	@echo "  ps          Show container status"
	@echo "  test        unit + integration tests"
	@echo "  test-unit   unit tests only (no Docker required)"
	@echo "  test-int    integration tests only (stack must be running)"
	@echo "  smoke       End-to-end smoke test"
	@echo "  backup      Hard-link snapshot of live /var/www/ClinicDx"
	@echo "  prefetch    Pre-download model/KB artifacts into volumes"
	@echo "  lint        ruff + mypy"
