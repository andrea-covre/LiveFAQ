import argparse
from kafka import KafkaConsumer
import time

QUESTION_FILE_NAME = "incoming_questions.tmp"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                        description = 'LiveFAQ consumer consumes questions from Kafa.')

    parser.add_argument("-t", "--timeout", type = int,
                            default = 10000,
                            choices = [10000, 20000],
                            help = "Specify number of clusters")

    args = parser.parse_args()
    timeout_ms = args.timeout

    topic_name = str("question-events")
    kafka_consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest',
                                 bootstrap_servers=['localhost:9092'], api_version=(0, 10), consumer_timeout_ms=timeout_ms)
    print("Listening For Questions (Timeout set to " + str(timeout_ms) + " ms)")
    questions_received_so_far = []
    for msg in kafka_consumer:
        incoming_question = str(msg.value.decode())
        incoming_question = incoming_question.lower()
        print ("Received: " + incoming_question)
        questions_received_so_far.append(incoming_question)
        with open(QUESTION_FILE_NAME,'w') as f:
            for question in questions_received_so_far:
                f.write(f"{question}\n")

    kafka_consumer.close()
