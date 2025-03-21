.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	pytest -v --color=yes tests/

.PHONY : format
format :
	isort .
	black .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
