import falcon
from api import SimpleClassifier, ExpandedClassifier

api = falcon.API()
simple_classifier = SimpleClassifier()
expanded_classifier = ExpandedClassifier()
api.add_route('/classify', simple_classifier)
api.add_route('/expanded-classify', expanded_classifier)
