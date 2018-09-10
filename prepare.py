import numpy as np
import random
import json
import argparse
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from utils import User
from tqdm import tqdm

from multiprocessing import Pool


users = {}
db_config = {}
limit = 5


def get_retweet_user(inputs):
    """
    This function searches engaged users which re-tweet the original tweet(tweet_id)
    :param inputs: tuple which have tweet_id and rating(True/False)
    :return:
    """
    tweet_id, rating = inputs

    client = MongoClient(**db_config)
    db: Database = client['fakenews']
    tweets_collections: Collection = db['tweets_sample']

    query = tweets_collections.find({"retweeted_status.id": tweet_id}).sort([("created_at", 1)]).limit(limit)
    engaged_users = []

    for tweet in query:
        engaged_users.append(tweet['user'])

    if len(engaged_users) == 0:
        return [], False

    if len(engaged_users) < limit:
        engaged_users.extend([random.choice(engaged_users) for i in range(limit - len(engaged_users))])
        engaged_users.sort(key=lambda x: x['created_at'])

    engaged_users_vector = [User(user).vector for user in engaged_users]

    if rating:
        rating = 1
    else:
        rating = 0

    return engaged_users_vector, rating


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config file path", required=True)
    parser.add_argument('--max-length', type=int, help="max sequence length", required=True)
    parser.add_argument('--output', type=str, help="output file path", required=True)
    args = parser.parse_args()

    limit = args.max_length

    config = json.load(open(args.config, "rt", encoding="utf8"))
    db_config.update(config['db_config'])
    client = MongoClient(**db_config)
    db: Database = client['fakenews']
    tweets_collections: Collection = db['tweets_sample']
    users_collection: Collection = db['users']

    for user in users_collection.find({"rating": {"$ne": None}}):
        users[user['_id']] = user

    print("[+] Load users")
    print("[+] Total users given labels: {:,}".format(len(users)))

    unique_tweets_filter = {
        "retweeted_status": {
            "$exists": False
        },
        "in_reply_to_status_id": {
            "$eq": None
        }
    }

    tqdm.write("[+] Total tweets: {:,}".format(tweets_collections.count_documents(unique_tweets_filter)))

    tweets = []

    # Get original tweets except for re-tweets and rely tweets
    loader = tweets_collections.find(unique_tweets_filter, {'_id': 1, 'user': 1})
    # Filter out users whether each user has a ground truth label
    loader = map(lambda x: (x['_id'], x['user']['id'], users.get(x['user']['id'], {}).get('rating', None)), loader)
    loader = filter(lambda x: x[2] is not None, loader)

    # For Debug
    # TODO: Make it parallel
    with open(args.output, "wt") as output_f, tqdm(desc="tweets") as pbar:
        for i, (tweet_id, user_id, rating) in enumerate(loader):
            pbar.update()

            vector, rating = get_retweet_user((tweet_id, rating))
            if vector:
                vector = np.array(vector)
                json_data = json.dumps([vector.tolist(), rating])
                output_f.write(json_data + "\n")
