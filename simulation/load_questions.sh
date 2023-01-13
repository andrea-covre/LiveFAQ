#!/bin/bash
echo "File Being Loaded: $1"
python3 ../kafka_producer.py <<EOF
$(while read line
do 
    echo $line
done < $1)
quit
EOF