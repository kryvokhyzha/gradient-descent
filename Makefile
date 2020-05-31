docker_run: 
	docker run -p ${PORT}:${PORT} --rm -it gradient-descent
docker_build: 
	docker build -t gradient-descent .