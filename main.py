import falcon
from api import Classifier

api = falcon.API()
classifier = Classifier()
api.add_route('/classify', classifier)
