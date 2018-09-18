import argparse
import json
import random
from multiprocessing import Pool

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from tqdm import tqdm

from utils import User

users = {}
db_config = {}
limit = 5


def get_retweet_user(inputs):
    """
    This function searches engaged users which re-tweet the original tweet(tweet_id)
    :param inputs: tuple which have tweet_id and rating(True/False)
    :return: tuple(engaged_user_vector: np.array, rating: 1/0)
    """
    tweet_id, user_id, tweet_text, rating = inputs

    client = MongoClient(**db_config)
    db: Database = client['fakenews']
    tweets_collections: Collection = db['tweets']

    query = tweets_collections.find({"retweeted_status.id": tweet_id}).sort([("created_at", 1)]).limit(limit)
    engaged_users = []

    for tweet in query:
        engaged_users.append(tweet['user'])

    if len(engaged_users) == 0:
        return [], False, False

    if len(engaged_users) < limit:
        engaged_users.extend([random.choice(engaged_users) for i in range(limit - len(engaged_users))])
        engaged_users.sort(key=lambda x: x['created_at'])

    engaged_users_vector = [User(user).vector for user in engaged_users]

    if rating:
        rating = 1
    else:
        rating = 0

    return engaged_users_vector, tweet_text, rating


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config file path", required=True)
    parser.add_argument('--max-length', type=int, help="max sequence length", required=True)
    parser.add_argument('--output', type=str, help="output file path", required=True)
    parser.add_argument('-c', '--cpu', type=int, default=20, metavar="20", help="amount of cpu for multiprocessing")
    args = parser.parse_args()

    limit = args.max_length

    config = json.load(open(args.config, "rt", encoding="utf8"))
    db_config.update(config['db_config'])
    client = MongoClient(**db_config)
    db: Database = client['fakenews']
    tweets_collections: Collection = db['tweets']
    users_collection: Collection = db['users']

    for user in users_collection.find({"rating": {"$ne": None}}):
        users[user['_id']] = user

    print("[+] Load users")
    print("[+] Total users given labels: {:,}".format(len(users)))

    unique_tweets_filter = {
        "retweeted_status.id": {"$exists": False},
        "in_reply_to_status_id": {"$eq": None},
        "lang": {"$eq": "en"}
    }

    tweets_count = tweets_collections.count_documents(unique_tweets_filter)
    tqdm.write("[+] Total tweets: {:,}".format(tweets_count))

    # Get original tweets except for re-tweets and rely tweets
    loader = tweets_collections.find(unique_tweets_filter, {'_id': 1, 'user': 1, 'text': 1})
    # Filter out users whether each user has a ground truth label
    loader = map(lambda x: (x['_id'], x['user']['id'], x['text'], users.get(x['user']['id'], {}).get('rating', None)), loader)
    loader = filter(lambda x: x[3] is not None, loader)

    with open(args.output, "wt") as output_f, tqdm(desc="tweets", total=tweets_count) as pbar, Pool(args.cpu) as pool:
        for i, (vector, tweet, rating) in enumerate(pool.imap_unordered(get_retweet_user, loader)):
            pbar.update()
            if vector:
                vector = np.array(vector)
                json_data = json.dumps([vector.tolist(), tweet, rating])
                output_f.write(json_data + "\n")
