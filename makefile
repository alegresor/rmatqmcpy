doctests:
	pytest --doctest-modules manifoldqmcpy/ -W ignore

doctestsaccept:
	pytest --doctest-modules manifoldqmcpy/ -W ignore --accept