import json
import falcon
from knn import KNNClassifier

class Classifier(object):

    def on_post(self, req, resp):
        knn_classifier = KNNClassifier(req.media)
        _, grouped_results = knn_classifier.bandsize_classifier()
        resp.body = json.dumps(grouped_results, ensure_ascii=False)
        resp.status = falcon.HTTP_200