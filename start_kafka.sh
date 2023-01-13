# # curl -sSL https://raw.githubusercontent.com/bitnami/containers/main/bitnami/kafka/docker-compose.yml > docker-compose.yml
# mkdir -p ./kafka_data
# mkdir -p ./zookeeper_data
# # cat docker-compose.yml | sed 's/zookeeper_data/.\/zookeeper_data/g' | sed 's/kafka_data/.\/kafka_data/g' > tmp.yml
# # cat tmp.yml > docker-compose.yml
# # rm tmp.yml
# docker-compose up -d


cd kafka_2.13-3.3.1

# bin/zookeeper-server-start.sh config/zookeeper.properties

bin/kafka-server-start.sh config/server.properties