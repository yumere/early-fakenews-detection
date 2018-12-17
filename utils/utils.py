import numpy as np
from datetime import datetime


class User(object):
    def __init__(self, user):
        self.user = user
        super(User, self).__init__()

    @property
    def vector(self):
        return np.array([self.length_of_user_description,
                         self.length_of_username,
                         self.followers_count,
                         self.friends_count,
                         self.statuses_count,
                         self.registration_age,
                         self.is_verified,
                         self.is_geo_enabled])

    @property
    def length_of_user_description(self):
        # TODO: make it correctly
        if self.user['description']:
            return len(self.user['description'])
        else:
            return 0

    @property
    def length_of_username(self):
        # TODO: make it correctly
        return len(self.user['screen_name'])

    @property
    def followers_count(self):
        return self.user['followers_count']

    @property
    def friends_count(self):
        return self.user['friends_count']

    @property
    def statuses_count(self):
        return self.user['statuses_count']

    @property
    def registration_age(self):
        # TODO: make it correctly
        now = datetime.now()
        days = (now - self.user['created_at']).days

        return days

    @property
    def is_verified(self):
        # verified: 1
        # not verified: 0

        return 1 if self.user['verified'] else 0

    @property
    def is_geo_enabled(self):
        # enabled: 1
        # not enabled: 0
        return 1 if self.user['geo_enabled'] else 0


class bcolors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
