.PHONY: setup run test lint typecheck ui clean

VENV_DIR := .venv
PYTHON   := python3
STREAMLIT_APP := src/trendbot/ui/streamlit/app.py

# --- Setup ---
setup:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	@. $(VENV_DIR)/bin/activate && pip install -e ".[dev]" -q
	@mkdir -p data/raw data/metadata output
	@echo "Setup complete. Run 'make run' to start."

# --- Run ---
run:
	@if [ ! -d "$(VENV_DIR)" ]; then $(MAKE) setup; fi
	@. $(VENV_DIR)/bin/activate && streamlit run $(STREAMLIT_APP) --server.headless true

# --- Test ---
test:
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && pytest; \
	else \
		pytest; \
	fi

# --- Lint ---
lint:
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && ruff check .; \
	else \
		ruff check .; \
	fi

# --- Typecheck ---
typecheck:
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && mypy src; \
	else \
		mypy src; \
	fi

# --- UI (legacy alias) ---
ui: run

# --- Clean ---
clean:
	rm -rf $(VENV_DIR) .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data output
