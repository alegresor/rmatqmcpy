doctests:
	pytest --doctest-modules flagqmcpy/ -W ignore

doctestsaccept:
	pytest --doctest-modules flagqmcpy/ -W ignore --accept

mkdocs_serve:
	cp README.md docs/index.md & mkdocs serve --livereload