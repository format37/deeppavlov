###
# this shell script uploads video file to container
###

# get id of running container with image name "nlp_transformers"
container_id=$(docker ps | grep nlp_transformers | awk '{print $1}')
echo "container id: $container_id"
# connect to container
sudo docker exec -it $container_id /bin/bash
