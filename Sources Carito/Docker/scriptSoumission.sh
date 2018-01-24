#  Monter image
sudo docker build -t  docker.synapse.org/syn11051567/test2:test6 .


# Tester image
sudo docker run docker.synapse.org/syn11051567/test2:test6 ./score_sc2.sh



# Connexion
 sudo  docker login docker.synapse.org

# Listing images
sudo docker images


# Envoyer docker
sudo docker push docker.synapse.org/syn11051567/test2:test6

