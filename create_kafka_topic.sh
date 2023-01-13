topic_name=$1
if [[ -z ${topic_name} ]]; then
    topic_name="question-events"
fi

echo "Topic Name: "${topic_name}

cd kafka_2.13-3.3.1

bin/kafka-topics.sh --create --topic ${topic_name} --bootstrap-server localhost:9092