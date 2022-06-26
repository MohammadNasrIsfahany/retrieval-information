

from typing import List,Tuple
import numpy as np
from math import log
from indexer_plus.inverted_index import InvertedIndex
from indexer_plus.constructor import BSBIIndex
from indexer_plus.text_preprocess import TextCleaner
from indexer_plus.utils import DocInfo
from indexer_plus.constructor import BSBIIndex
from indexer_plus.text_preprocess import TextCleaner
from indexer_plus.utils import DocInfo
import math
from indexer_plus.constructor import BSBIIndex
from indexer_plus.text_preprocess import TextCleaner
from indexer_plus.utils import DocInfo

class BM25:
    text_cleaner = TextCleaner()
    def __init__(self):
        self.b = 0.75
        self.k1 = 2
        self.index = BSBIIndex(data_dir='./Dataset_IR/Train', output_dir='./Output/')
        self.index.load()
        self.total_doc_num = len(self.index.doc_id_map)
        self.total_term_num = len(self.index.term_id_map)
        self.postings_dict = {}
        with InvertedIndex(self.index.index_name, postings_encoding=self.index.postings_encoding,
                           directory=self.index.output_dir) as inverted_index:
            self.postings_dict = inverted_index.postings_dict

    def read_train_files(self,dataset_dir):

        '''
            Description:
                This function reads train files

        '''
        self.index = BSBIIndex(data_dir=dataset_dir, output_dir='./Output/')
        self.index.load()
        # return self

    # def build_idf(self, loaded_index):
    #     self.total_doc_num = len(loaded_index.doc_id_map)
    #     self.total_term_num = len(loaded_index.term_id_map)
    #     self.postings_dict = {}
    #     with InvertedIndex(loaded_index.index_name, postings_encoding=loaded_index.postings_encoding,
    #                        directory=loaded_index.output_dir) as inverted_index:
    #         self.postings_dict = inverted_index.postings_dict 
    #     self.termsID = loaded_index.term_id_map
    #     self.idf = {}
    
    def get_term_idf(self, term=None):
        '''
            Description:
                This function return idf value of a term

        '''

        if not isinstance(term, int):
            term = self.termsID[term]
        
        start_posting_pointer, posting_list_len, bytes_num = self.postings_dict[term]
        return np.log(((self.total_doc_num - posting_list_len + 0.5) / (posting_list_len + 0.5)))
        # number of ducuments that have term 'term'
        # raise NotImplementedError
        
    def get_total_score(self, q, query_vec, d, doc_vec):
        f = dict()
        for t in query_vec:
            if t in doc_vec:
                # f[t] += doc_vec[t]
                f[t] = f.get(t, 0) + doc_vec[t]
        score = sum([query_vec[t] * ((f[t] * (self.k1 + 1))
                                     / (f[t] + self.k1 *
                                        (1 - self.b + self.b * (self.doc_info[d]['length'] / self.avg_length))))
                     for t in query_vec.keys()])
        return score

    def get_query_vector(self, q):
        query_vec = {}

        query_dic = {}
        for term, tf in q.items():
            query_dic[self.index.term_id_map[term]] = tf
        for term, tf in query_dic.items():
            term_idf = self.get_term_idf(term)
            query_vec[term] = term_idf
        return query_vec

    def get_doc_vector(self, q, d, doc_weight_scheme=None):
        doc_vec = {}

        query_vec = self.get_query_vector(q)
        doc_dict_info = self.doc_info[d]['terms']

        for qv in query_vec:
            if qv in doc_dict_info:
                if qv not in doc_vec:
                    doc_vec[qv] = (doc_dict_info[qv] + 1) * self.get_term_idf(qv)
            else:
                doc_vec[qv] = 1
        doc_vec = self.normalize_doc_vec(q, d, doc_vec)
        return doc_vec

    def normalize_doc_vec(self, q, d, doc_vec):

        x = 0
        for item in doc_vec:
            x += (doc_vec[item] * doc_vec[item])
        for item in doc_vec:
            doc_vec[item] = doc_vec[item] / math.sqrt(x)
        return doc_vec
    
    def calc_avg_length(self):
        # get from json files and calculate the avg
        len_sum = 0
        docinfo_len = len(self.index.doc_id_map)
        for i in range(docinfo_len):
            len_sum += self.doc_info[i]['length']
        avg = len_sum / docinfo_len
        return docinfo_len, avg
    
    def calculate_score(self, query, document):
        '''
            Description:
                This function calculates score of each doc according to
                the query 'query'

        '''
        # self.idf = idf
        # self.index = index


        self.text_cleaner = TextCleaner()
        self.doc_info = DocInfo(self.index.term_id_map)

        self.N, self.avg_length = self.calc_avg_length()

        self.default_query_weight_scheme = {"tf": 'b', "df": 't', "norm": None}  # boolean, idf, none
        self.default_doc_weight_scheme = {"tf": 'n', "df": 'n', "norm": None}  # natural, none

        # self.query_weight_scheme = query_weight_scheme if query_weight_scheme is not None \
        #     else self.default_query_weight_scheme  # Modified (added)
        # self.doc_weight_scheme = doc_weight_scheme if doc_weight_scheme is not None \
        #     else self.default_doc_weight_scheme  # Modified (added)
        query_vec = self.get_query_vector(query)
        norm_doc_vec = self.get_doc_vector(query, document)
        return self.get_total_score(query, query_vec, document, norm_doc_vec)

        raise NotImplementedError

    def get_similar_docs(self, query: str)-> list:
        # self.index = index
        '''
            Description:
                This function gets a query and ranks the dataset based on BM25 score sort by score
            output: a list of dicts [{"text": "document 1", "bm25_score": 1},
                                     {"text": "document 2", "bm25_score": 0.8}]
        '''
        query_dict = self.text_cleaner.tokenize(query)
        related_docs = self.index.retrieve(query_dict)

        doc_score = list()
        for doc_id in related_docs:
            doc_score.append((doc_id, self.calculate_score(query_dict, doc_id)))
        # print(doc_score)
        return self.get_sorted_result(doc_score)
        raise NotImplementedError

    def get_sorted_result(self, doc_score):
        sorted_list = sorted(doc_score, key=lambda item: item[1], reverse=True)
        print(sorted_list)
        return [self.index.doc_id_map[doc[0]] for doc in sorted_list]
    
def read_query():
    return input('query >> ')

if __name__ == "__main__":
    dataset_dir = "./Dataset_IR/Train"

    bm25 = BM25().read_train_files(dataset_dir)
    while True:
        query = read_query()
        # print(query)
        print(BM25().get_similar_docs(query))


