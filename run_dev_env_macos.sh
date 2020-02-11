export DISPLAY=$DISPLAY

docker-compose down && docker-compose build;
docker-compose up;