doctests:
	pytest --doctest-modules rmatqmcpy/ -W ignore

doctestsaccept:
	pytest --doctest-modules rmatqmcpy/ -W ignore --accept

mkdocs_serve:
	cp README.md docs/index.md & mkdocs serve --livereload