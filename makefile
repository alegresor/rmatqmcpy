doctests:
	pytest --doctest-modules flagqmcpy/ -W ignore

doctestsaccept:
	pytest --doctest-modules flagqmcpy/ -W ignore --accept