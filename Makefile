SHELL=/bin/bash

test:
	python -m unittest

jupyter: build-image
	docker run -t -i -p 8888:8888 -v $$(pwd):/app -e PYTHONPATH=/app -v $$(pwd)/notebooks:/data statwolf/llf:0.0.0

freeze:
	pip freeze | grep -v tensorflow > deps.txt

_develop:
	while inotifywait -r -e create,modify ./modules --exclude ''\\.pyc$$''; do make test; done

develop:
	docker-compose up --build

build-image:
	docker-compose build

base-image:
	cd base-image && make build && cd ..

.PHONY: base-image develop build-image jupyter