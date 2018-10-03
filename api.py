import json
import falcon
from knn import KNNClassifier

class SimpleClassifier(object):

    def on_post(self, req, resp):
        knn_classifier = KNNClassifier(req.media)
        _, grouped_results, _ = knn_classifier.bandsize_classifier()
        resp.body = json.dumps(grouped_results, ensure_ascii=False)
        resp.status = falcon.HTTP_200

class ExpandedClassifier(object):

    def on_post(self, req, resp):
        knn_classifier = KNNClassifier(req.media)
        _, grouped_results, _ = knn_classifier.bandsize_classifier()
        mu, CoV, k = knn_classifier.get_statistics()
        response_object = {
            "statistics": {
                'mu': mu,
                'CoV': CoV,
                'k': k
            },
            "samples": grouped_results
        }
        resp.body = json.dumps(response_object, ensure_ascii=False)
        resp.status = falcon.HTTP_200
