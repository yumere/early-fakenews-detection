try:
    import ujson as json
except ImportError:
    import json
import argparse
import math
import random
from multiprocessing import Pool
from typing import List

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from tqdm import tqdm

from utils import User
from utils import bcolors


def get_retweet_user(inputs):
    """
    This function searches engaged users which re-tweet the original tweet(tweet_id)
    :param inputs: tuple which have tweet_id and rating(True/False)
    :return: tuple(engaged_user_vector: np.array, rating: 1/0)
    """
    chunk, labeled_users, db_config, limit = inputs
    results = []

    client = MongoClient(**{
        'host': db_config['host'],
        'port': db_config['port']
    })
    db: Database = client[db_config['db']]
    tweets_collections: Collection = db[db_config['collection']]

    for tweet_id, user_id, tweet_text, rating, created_at in chunk:
        query = tweets_collections.find({"retweeted_status.id": tweet_id}).sort([("created_at", 1)]).limit(limit)
        engaged_users = []

        for tweet in query:
            engaged_users.append(tweet['user'])

        if len(engaged_users) == 0:
            continue

        if len(engaged_users) < limit:
            engaged_users.extend([random.choice(engaged_users) for i in range(limit - len(engaged_users))])
            engaged_users.sort(key=lambda x: x['created_at'])

        engaged_users_vector = [User(user).vector.tolist() for user in engaged_users]

        if rating:
            rating = 1
        else:
            rating = 0

        if engaged_users_vector:
            results.append((engaged_users_vector, tweet_text, rating, user_id, str(created_at)))

    return results


def chunk_data(data: List[str], n_processes: int) -> List[List[str]]:
    chunk_size = int(math.ceil(len(data) / n_processes))
    for i in range(0, len(data), chunk_size):
        yield data[i: i + chunk_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config file path", required=True)
    parser.add_argument('--max-length', type=int, help="max sequence length", required=True)
    parser.add_argument('--output', type=str, help="output file path", required=True)
    parser.add_argument('-n', '--n_processes', type=int, default=20, metavar="20", help="amount of cpu for multiprocessing")
    args = parser.parse_args()

    limit = args.max_length

    config = json.load(open(args.config, "rt", encoding="utf8"))
    db_config = config['db_config']
    client = MongoClient(**{
        'host': db_config['host'],
        'port': db_config['port']
    })
    db: Database = client[db_config['db']]
    tweets_collections: Collection = db[db_config['collection']]
    users_collection: Collection = db['users']

    labeled_users = {}

    for user in users_collection.find({"rating": {"$ne": None}}):
        labeled_users[user['_id']] = user

    print(bcolors.OKGREEN + "[+] Load users" + bcolors.ENDC)
    print(bcolors.OKBLUE + "[+] Total labeled users: {:,}".format(len(labeled_users)) + bcolors.ENDC)

    unique_tweets_filter = {
        "retweeted_status.id": {"$exists": False},
        "in_reply_to_status_id": {"$eq": None},
        "lang": {"$eq": "en"}
    }

    tweets_count = tweets_collections.count_documents(unique_tweets_filter)
    tqdm.write(bcolors.OKBLUE + "[+] Total tweets: {:,}".format(tweets_count) + bcolors.ENDC)

    # Get original tweets except for re-tweets and rely tweets
    loader = tweets_collections.find(unique_tweets_filter, {'_id': 1, 'user': 1, 'text': 1, 'created_at': 1})
    # Filter out users whether each user has a ground truth label
    loader = map(lambda x: (x['_id'], x['user']['id'], x['text'], labeled_users.get(x['user']['id'], {}).get('rating', None), x['created_at']), loader)
    loader = filter(lambda x: x[3] is not None, loader)
    chunks = chunk_data(list(loader), args.n_processes)

    with tqdm(desc="tweets", total=tweets_count) as pbar, Pool(args.n_processes) as pool:
        for i, result in enumerate(pool.imap_unordered(get_retweet_user, [(chunk, labeled_users, db_config, limit) for chunk in chunks])):
            pbar.update(len(result))
            with open(args.output, "at", encoding="utf-8") as output:
                for r in result:
                    output.write(json.dumps(r) + "\n")
