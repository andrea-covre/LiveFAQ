from kafka import KafkaProducer
import time
import argparse

def publish_message(producer_instance, topic_name, key, value):
    try:
        key_bytes = bytes(key, encoding='utf-8')
        value_bytes = bytes(value, encoding='utf-8')
        producer_instance.send(topic_name, key=key_bytes, value=value_bytes)
        producer_instance.flush()
        print('Message published successfully.')
    except Exception as ex:
        print('Exception in publishing message')
        print(str(ex))


def connect_kafka_producer():
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10))
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return _producer

def get_cli_args():
    parser = argparse.ArgumentParser(
                        description = 'LiveFAQ producer publishes questions to Kafka topic.')

    parser.add_argument("-t", "--time", type = int, 
                            default = 0,
                            choices = range(1, 5),
                            help = "Time between questions being published, 0 if manual")

    parser.add_argument("-q", "--questions", type = str, 
                            default = "simulation/questions-1.txt",
                            choices = ["simulation/questions-" + str(n) + ".txt" for n in range(1, 6)],
                            help = "Specify txt file to ask questions from")

    args = parser.parse_args()
    return args.time, args.questions



topic_name = str("question-events")

if __name__ == "__main__":
    t, questions = get_cli_args()

    kafka_producer = connect_kafka_producer()

    message_ctr = 1

    if t == 0:
        while True:
            user_question = input("Enter your Question: ")

            if str(user_question) in ["exit", "quit", "q", "esc", "exit()", "quit()"]:
                break

            if kafka_producer is not None:
                print("Producer Connection Success!")
                publish_message(kafka_producer, topic_name, 'raw', str(user_question))
            else:
                print("Producer Connection Failed!")
    else:
        question_list = []
        with open(questions) as question_file:
            question_list = [line.rstrip() for line in question_file]
        for question in question_list:
            if kafka_producer is not None:
                print("Producer Connection Success!")
                publish_message(kafka_producer, topic_name, 'raw', str(question))
                time.sleep(t)
            else:
                print("Producer Connection Failed!")
                break
    if kafka_producer is not None:
        kafka_producer.close()