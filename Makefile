build:
	docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom .
push:
	docker push us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom

bash:
	docker run --rm -it us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom bash
test:
	docker run --rm -it us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom

