build:
	docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom .
push:
	docker push us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom

bash:
	docker run --gpus=all -p 8888:8080 --rm -it us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom bash
test:
	docker run --gpus=all -p 8888:8080 --rm -it us-central1-docker.pkg.dev/testloop-5b5ed/testloop/omniparser:custom

