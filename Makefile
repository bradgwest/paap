deps:
	pip install -Ur requirements_dev.txt

lint:
	isort --check-only **/*.py
	black --check **/*.py
	flake8 .

format:
	black **/*.py
	isort **/*.py
