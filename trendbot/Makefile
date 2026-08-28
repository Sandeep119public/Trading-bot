.PHONY: install lint typecheck test ui clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

ui:
	streamlit run src/trendbot/ui/streamlit/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache
