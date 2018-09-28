import time
import json
import ptvsd
from knn import KNNClassifier

print("Waiting to attach")

ADDRESS = ('0.0.0.0', 3000)
ptvsd.enable_attach(ADDRESS)
ptvsd.wait_for_attach()
time.sleep(0.2)

print("Attached")

with open('example_input_data.json') as data_file:
    loaded_data = json.load(data_file)

knn_classifier = KNNClassifier(loaded_data)
all_results = knn_classifier.bandsize_classifier()


print("Ending")
