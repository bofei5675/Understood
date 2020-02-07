# -*- mode: python; coding: utf-8; -*-
from recommendation import AbstractRecommender

import pandas as pd
from utils import time_popularity
import heapq

class Recommender(AbstractRecommender):
    def __init__(self):
        self.last_page_viewed = {}
        # 'url_path:{ last_row_num, popularity }'
        self.time_popularity = {}
        self.popularity_heap = []
        self.alpha = 0.1

    def observe(self, user_interaction):

        # update popularity
        url_path = user_interaction["URL_PATH"]
        row_number = user_interaction['ROW_NUM']
        if url_path in self.time_popularity:
            tmp = self.time_popularity[url_path]
            t = tmp[1]
            popularity = time_popularity(row_number, t, self.alpha)
            tmp[0] += popularity
            tmp[1] = row_number
            heapq.heapify(self.popularity_heap)
        else:
            popularity = time_popularity(row_number, row_number, self.alpha)
            tmp = [popularity, row_number, url_path]
            self.time_popularity[url_path] = tmp
            heapq.heappush(self.popularity_heap, tmp)

    def recommend(self, user_id, n):

        # based on time popularity
        result = heapq.nlargest(n, self.popularity_heap)
        return [i[2] for i in result]

    def train(self, user_interactions):
        for _, s in user_interactions.iterrows():
            self.observe(s)


if __name__ == "__main__":
    user_interactions = pd.read_csv('./train.csv')#.iloc[-10:]
    user_interactions = user_interactions.reindex(index=user_interactions.index[::-1])


    n = 3
    recommender = Recommender()

    # train
    print('Train')
    recommender.train(user_interactions)

    # test
    print('Test')
    user_interactions = pd.read_csv('./test.csv')#.iloc[-10:]
    user_interactions = user_interactions.reindex(index=user_interactions.index[::-1])
    total_num = 0
    right_prediction = 0
    for _, s in user_interactions.iterrows():
        recommendations = recommender.recommend(s["USER_ID"], n)
        recommender.observe(s)
        url_path = s['URL_PATH']
        if url_path in recommendations:
            right_prediction += 1
        total_num += 1
    print('Precision:', right_prediction/total_num)