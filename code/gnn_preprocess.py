import json, time, random
import os, sys, glob, csv, re, argparse
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from collections import defaultdict
import seaborn as sns
import matplotlib.pylab as plt

import nltk
# nltk.download('punkt')
import torch



class GCN_PreProcess():
    def __init__(self, config, gossipcop=False, politifact=False, pheme=False):
        self.config = config
        self.data_dir = config['data_dir']
        self.subset = config['subset']
        self.datasets = []
        if politifact:
            self.datasets.append('politifact')
        if gossipcop:
            self.datasets.append('gossipcop')
        if pheme:
            self.datasets.append('pheme')
        
        # if config['create_aggregate_folder']:
        #     self.create_aggregate_folder()
        # if config['create_adj_matrix']:
        #     self.create_adj_matrix()
        # if config['create_feat_matrix']:
        #     self.create_feat_matrix()
        # if config['create_labels']:
        #     self.create_labels()
        # if config['create_split_masks']:
        #     self.create_split_masks()
        
        # self.create_dicts()
        # self.create_adj_matrix()
        # # self.create_adj_matrix_exclusive()
        # self.create_labels()
        # self.create_random_split_masks()
        # self.create_user_splits_list()
        # self.create_feat_matrix()   
        
        
        # self.check_overlapping_users()
        # self.generate_graph_stats()
        self.create_filtered_follower_following()
        
        



    def create_aggregate_folder(self):
        print("\nCreating aggregate files for all docs and their users......")
        for dataset in self.datasets:
            print("\n" + "-"*60 + "\n \t\t Analyzing {} dataset\n".format(dataset) + '-'*60)
            if dataset == 'pheme':
                src_dir = os.path.join(self.data_dir, 'base_data', 'pheme_cv')
                events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
                c=0
                docs_done = defaultdict(list)
                for fold, event in enumerate(events):
                    print("\nIterating over {}...".format(event))
                    data_dir = os.path.join(src_dir, event)
                    for root, dirs, files in os.walk(data_dir):
                        for file in files:
                            if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                                continue
                            else:
                                src_file_path = os.path.join(root, file)
                                src_file = json.load(open(src_file_path, 'r'))
                                user_id = int(src_file['user']['id'])
                                doc = root.split('\\')[-2]
                                docs_done[doc].append(user_id)
                                c+=1
                                if c%2000 == 0:
                                   print("{} done...".format(c))
                                   
                print("\nTotal tweets/re-tweets in the data set = ", c)
                print("\nWriting all the info in the dir..")
                for doc, user_list in docs_done.items():
                    write_file = './data/complete_data/pheme/complete/{}.json'.format(doc)
                    with open(write_file, 'w+') as j:
                        temp_dict = {}
                        temp_dict['users'] = list(set(user_list))
                        json.dump(temp_dict, j)
                print("\nDONE..!!")
                
            else:
                user_contexts = ['tweets', 'retweets']
                docs_done = defaultdict(set)
                for user_context in user_contexts:
                    print("\nIterating over : ", user_context)
                    src_dir = os.path.join(self.data_dir, 'complete_data', dataset, user_context)  
                    dest_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
                    if not os.path.exists(dest_dir):
                        print("Creating dir:  {}\n".format(dest_dir))
                        os.makedirs(dest_dir)
                    if user_context == 'tweets':
                        for root, dirs, files in os.walk(src_dir):
                            for count, file in enumerate(files):
                                doc = file.split('.')[0]
                                src_file_path = os.path.join(root, file)
                                src_file = pd.read_csv(src_file_path)
                                user_ids = src_file['user_id']
                                user_ids = [s for s in user_ids if isinstance(s, int)]
                                user_ids = list(set(user_ids))
                                docs_done[doc].update(user_ids)
                                if count==1:
                                    print(doc, docs_done[doc])
                                if count%2000 == 0:
                                    print("{} done".format(count))
                        
                    elif user_context == 'retweets':
                        if not os.path.exists(dest_dir):
                            print("Creating dir   {}", dest_dir)
                            os.makedirs(dest_dir)
                        for root, dirs, files in os.walk(src_dir):
                            for count,file in enumerate(files):
                                doc = file.split('.')[0]
                                src_file_path = os.path.join(root, file)
                                with open(src_file_path, encoding= 'utf-8', newline = '') as csv_file:
                                    lines = csv_file.readlines()
                                    for line in lines:
                                        file = json.loads(line)
                                        docs_done[doc].update([file['user']["id"]])
                                if count==1:
                                    print(doc, docs_done[doc])
                                if count%2000 == 0:
                                    print("{} done".format(count))
                print("\nWriting into files at:   ", dest_dir)
                for doc, user_list in docs_done.items():
                    dest_file_path = os.path.join(dest_dir, str(doc)+'.json')
                    with open(dest_file_path, 'w+') as j:
                        temp_dict = {}
                        temp_dict['users'] = list(user_list)
                        json.dump(temp_dict, j)
        return None
    
    
    
    
    def create_dicts(self):
        
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t\t   Creating dicts for  {} dataset \n".format(dataset) + '-'*100)
            all_users = set()
            all_docs = set()
            doc2id, user2id = {}, {}
            doc2tags = {}
            
            if dataset == 'pheme':
                src_dir = os.path.join(self.data_dir, 'base_data', 'pheme_cv')
                
                print("\nCreating doc2id, doc2tags and user2id dicts....")
                start = time.time()
                # Getting all users and docs from the tweets-retweets data
                count=0
                for root, dirs, files in os.walk(src_dir):
                    for file in files:
                        if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                            continue
                        else:
                            src_file_path = os.path.join(root, file)
                            src_file = json.load(open(src_file_path, 'r'))
                            user_id = int(src_file['user']['id'])
                            doc_key = root.split("\\")[-2]
                            all_users.update([user_id])
                            all_docs.update([doc_key])
                            doc2id[str(doc_key)] = len(all_docs) -1
                            text = src_file['text'].lower()
                            hashtags = {tag.strip("#") for tag in text.split() if tag.startswith("#")}
                            doc2tags[str(doc_key)] = list(hashtags)
                            count+=1
                            
            else:
                src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
                print("\nCreating doc2id, doc2tags and user2id dicts....")
                start = time.time()
                temp_list = []
                restricted_users = json.load(open('./data/complete_data/restricted_users_{}.json'.format(dataset), 'r'))
                done_users = json.load(open('./data/complete_data/{}/done_users.json'.format(dataset), 'r'))
                # Getting all users and docs from the tweets-retweets data
                count=0
                user_contexts = ['fake', 'real']
                for user_context in user_contexts:
                    data_dir = os.path.join(self.data_dir, 'complete_data', dataset, user_context)
                    for root, dirs, files in os.walk(data_dir):
                        for count,file in enumerate(files):
                            doc= root.split('\\')[-1]
                            temp_list.append(str(doc))
                print("Len of total docs list = ", len(temp_list)) 
                avg_num_users = []
                for root, dirs, files in os.walk(src_dir):
                    for count, file in enumerate(files):
                        doc_key = file.split('.')[0]
                        if str(doc_key) in temp_list:
                            src_file_path = os.path.join(root, file)
                            src_file = json.load(open(src_file_path, 'r'))
                            users = src_file['users']
                            # users = [s for s in users if isinstance(s, int) and str(s) not in restricted_users['restricted_users']]
                            users = [s for s in users if isinstance(s, int) and s in done_users['done_users']]
                            # users = [s for s in users if isinstance(s, int)]
                            all_users.update(users[:self.subset])
                            avg_num_users.append(len(users))
                            all_docs.update([doc_key])
                            doc2id[str(doc_key)] = len(all_docs) -1
                    
                
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))

            all_users, all_docs = list(all_users), list(all_docs)
            num_users, num_docs = len(all_users), len(all_docs)
            print("\nTotal no. of docs in the dataset = {} \nTotal no. of unique users in this dataset  = {}".format(num_docs, num_users))
            print("Avg no. of users in each doc = {}".format(sum(avg_num_users)/len(avg_num_users)))
            
            for count, user in enumerate(all_users):
                user2id[str(user)] = count + num_docs
            
            node2id = doc2id.copy()
            node2id.update(user2id)
            print("node2id size = ", len(node2id))
            
             
            doc_file = self.data_dir+ '/complete_data' + '/doc2id_{}.json'.format(dataset)
            user_file = self.data_dir+ '/complete_data' + '/user2id_{}.json'.format(dataset)
            node_file = self.data_dir + '/complete_data' + '/node2id_{}.json'.format(dataset)
            # hash_file = self.data_dir+ '/complete_data' + '/doc2tags_{}.json'.format(dataset)
            # print("\nSaving doc2id dict in  {}\nSaving user2id dict in   {} \ndoc2tags dict in  {}".format(doc_file, user_file, hash_file))
            with open(doc_file, 'w+') as json_file:
                json.dump(doc2id, json_file)
            with open(user_file, 'w+') as json_file:
                json.dump(user2id, json_file)
            with open(node_file, 'w+') as json_file:
                json.dump(node2id, json_file)
            # with open(hash_file, 'w+') as json_file:
            #     json.dump(doc2tags, json_file)
        
        return None
                    
                    
                    
                    
                    
    def create_adj_matrix(self):
        
        for dataset in self.datasets:
            with open('./data/complete_data/user2id_{}.json'.format(dataset),'r') as j:
                   user2id = json.load(j)
            with open('./data/complete_data/doc2id_{}.json'.format(dataset),'r') as j:
                   doc2id = json.load(j)
            print("\n\n" + "-"*100 + "\n \t\t\t   Analyzing  {} dataset for adj_matrix\n".format(dataset) + '-'*100)
            # src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
            src_dir = os.path.join(self.data_dir, 'base_data', 'pheme_cv')
            
            num_users, num_docs = len(user2id), len(doc2id)
            print("\nNo.of unique users = ", num_users)
            print("No.of docs = ", num_docs)
            
            # Creating the adjacency matrix (doc-user edges)
            adj_matrix = lil_matrix((num_docs+num_users, num_users+num_docs))
            # adj_matrix = np.zeros((num_docs+num_users, num_users+num_docs))
            # adj_matrix_file = './data/complete_data/adj_matrix_pheme.npz'
            # adj_matrix = load_npz(adj_matrix_file)
            adj_matrix = lil_matrix(adj_matrix)
            # Creating self-loops for each node (diagonals are 1's)
            for i in range(adj_matrix.shape[0]):
                adj_matrix[i,i] = 1
            print_iter = int(num_docs/10)
            print("\nSize of adjacency matrix = {} \nPrinting every  {} docs".format(adj_matrix.shape, print_iter))
            start = time.time()
            
            
            if dataset == 'pheme':
                print("\nPreparing entries for doc-user and doc-doc pairs...")
                with open('./data/complete_data/doc2tags_{}.json'.format(dataset),'r') as j:
                   doc2tags = json.load(j)
                src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
                for root, dirs, files in os.walk(src_dir):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        doc_key = file.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        users = src_file['users']
                        for user in users:   
                            adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                            adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                        
                        # Add edges between docs that use same hashtags as the current one
                        common_tag_docs = []
                        user_contexts = ['user_followers_filtered', 'users_following_filtered']
                        for doc in doc2tags.keys():
                            if doc==doc_key:
                                continue
                            source_doc_tags = set(doc2tags[str(doc_key)])
                            target_doc_tags = set(doc2tags[str(doc)])
                            if len(source_doc_tags.intersection(target_doc_tags)) >0:
                                common_tag_docs.append(doc)
                                
                        for common_doc in common_tag_docs:
                            adj_matrix[doc2id[str(doc_key)], doc2id[str(common_doc)]] = 1
                            adj_matrix[doc2id[str(common_doc)], doc2id[str(doc_key)]] = 1
                            
                            common_doc_file = os.path.join(root, str(common_doc)+'.json')
                            common_doc_file = json.load(open(common_doc_file, 'r'))
                            common_tag_users = common_doc_file['users']
                            for user in users: 
                                for common_tag_user in common_tag_users:
                                    adj_matrix[user2id[str(common_tag_user)], user2id[str(user)]] = 1
                                    adj_matrix[user2id[str(user)], user2id[str(common_tag_user)]] = 1
                            
                            
                            # for context in user_contexts:
                            #     common_doc_users_file = os.path.join(root, str(common_doc)+'.json')
                            #     common_doc_users = json.load(open(common_doc_users_file, 'r'))
                            #     users = common_doc_users['users']
                            #     for user in users:
                            #         additional_users_file = './data/complete_data/pheme/'+context+str(user)+'.json'
                            #         add_users = json.load(open(additional_users_file, 'r'))
                            #         followers = add_users['followers'] if context == 'user_followers_filtered' else add_users['following']   
                            #         followers = list(map(int, followers))
                            #         for follower in followers:
                            #             if follower in all_users:
                            #                 adj_matrix[doc2id[str(doc_key)], user2id[str(follower)]]=1
                            #                 adj_matrix[user2id[str(follower)], doc2id[str(doc_key)]]=1
                                
                        if count%print_iter==0:
                            print("{} / {} done..".format(count+1, num_docs))
            
            else:
                print("\nPreparing entries for doc-user pairs...")
                src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
                not_found=0
                for root, dirs, files in os.walk(src_dir):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        doc_key = file.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        users = src_file['users']
                        for user in users:  
                            if str(doc_key) in doc2id and str(user) in user2id:
                                adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                                adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                            else:
                                not_found+=1


            end = time.time() 
            hrs, mins, secs = self.calc_elapsed_time(start, end)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))  
            print("Not Found users = ", not_found)
            print("Non-zero entries = ", adj_matrix.getnnz())
            # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            # Creating the adjacency matrix (user-user edges)
            user_contexts = ['user_followers_filtered', 'user_following_filtered']
            start = time.time()
            key_errors, not_found, overlaps = 0,0,0
            print("\nPreparing entries for user-user pairs...")
            print_iter = int(num_users/10)
            print("Printing every {}  users done".format(print_iter))
            
            for user_context in user_contexts:
                print("    - from {}  folder...".format(user_context))
                src_dir2 = os.path.join(self.data_dir, 'complete_data', dataset, user_context)
                for root, dirs, files in os.walk(src_dir2):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        # user_id = src_file_path.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        user_id = int(src_file['user_id'])
                        if str(user_id) in user2id:
                            followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file['following']   
                            followers = list(map(int, followers))
                            for follower in followers:
                                if str(follower) in user2id:
                                    adj_matrix[user2id[str(user_id)], user2id[str(follower)]]=1
                                    adj_matrix[user2id[str(follower)], user2id[str(user_id)]]=1
                                    
                        else:
                            not_found +=1
                        if count%print_iter==0:
                            # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                            print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, len(np.nonzero(adj_matrix)[0])))
                                         
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            # print("Key errors = ", key_errors)
            # print("Overlaps = ", overlaps)
            print("Not found user_ids = ", not_found)
            print("Total Non-zero entries = ", adj_matrix.getnnz())
            # print("Total Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            filename = self.data_dir+ '/complete_data' + '/adj_matrix_{}.npz'.format(dataset)
            # filename = self.data_dir+ '/complete_data' + '/adj_matrix_{}.npy'.format(dataset)
            print("\nMatrix construction done! Saving in  {}".format(filename))
            save_npz(filename, adj_matrix.tocsr())
            # np.save(filename, adj_matrix)
            
            # Creating an edge_list matrix of adj_matrix as required by some GCN frameworks
            print("\nCreating edge_index format of adj_matrix...")
            start = time.time()
            # G = nx.DiGraph(adj_matrix.tocsr())
            # temp_matrix = adj_matrix.toarray()
            # rows, cols = np.nonzero(temp_matrix)
            rows, cols = adj_matrix.nonzero()
            
            
            edge_index = np.vstack((np.array(rows), np.array(cols)))
            print("Edge index shape = ", edge_index.shape)
            
            edge_matrix_file = self.data_dir+ '/complete_data' + '/adj_matrix_{}_edge.npy'.format(dataset)
            print("saving edge_list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            # nx.write_edgelist(G, edge_matrix_file)
            # num_edges = len(G.edges())
            # print("No. of edges =  ", num_edges)           
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        return None
    
    
    
    def create_adj_matrix_exclusive(self):
        
        for dataset in self.datasets:
            with open('./data/complete_data/user2id_{}.json'.format(dataset),'r') as j:
                   user2id = json.load(j)
            with open('./data/complete_data/doc2id_{}.json'.format(dataset),'r') as j:
                   doc2id = json.load(j)
            print("\n\n" + "-"*100 + "\n \t\t\t   Analyzing  {} dataset for adj_matrix\n".format(dataset) + '-'*100)
            # src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
            src_dir = os.path.join(self.data_dir, 'base_data', 'pheme_cv')
            
            num_users, num_docs = len(user2id), len(doc2id)
            print("\nNo.of unique users = ", num_users)
            print("No.of docs = ", num_docs)
            
            # Creating the adjacency matrix (doc-user edges)
            adj_matrix = lil_matrix((num_docs+num_users, num_users+num_docs))
            # adj_matrix = np.zeros((num_docs+num_users, num_users+num_docs))
            # adj_matrix_file = './data/complete_data/adj_matrix_pheme.npz'
            # adj_matrix = load_npz(adj_matrix_file)
            # adj_matrix = lil_matrix(adj_matrix)
            # Creating self-loops for each node (diagonals are 1's)
            for i in range(adj_matrix.shape[0]):
                adj_matrix[i,i] = 1
            print_iter = int(num_docs/10)
            print("\nSize of adjacency matrix = {} \nPrinting every  {} docs".format(adj_matrix.shape, print_iter))
            start = time.time()
            
            
            if dataset == 'pheme':
                print("\nPreparing entries for doc-user and doc-doc pairs...")
                with open('./data/complete_data/doc2tags_{}.json'.format(dataset),'r') as j:
                   doc2tags = json.load(j)
                src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
                for root, dirs, files in os.walk(src_dir):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        doc_key = file.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        users = src_file['users']
                        for user in users:   
                            adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                            adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                        
                        # Add edges between docs that use same hashtags as the current one
                        common_tag_docs = []
                        user_contexts = ['user_followers_filtered', 'users_following_filtered']
                        for doc in doc2tags.keys():
                            if doc==doc_key:
                                continue
                            source_doc_tags = set(doc2tags[str(doc_key)])
                            target_doc_tags = set(doc2tags[str(doc)])
                            if len(source_doc_tags.intersection(target_doc_tags)) >0:
                                common_tag_docs.append(doc)
                                
                        for common_doc in common_tag_docs:
                            adj_matrix[doc2id[str(doc_key)], doc2id[str(common_doc)]] = 1
                            adj_matrix[doc2id[str(common_doc)], doc2id[str(doc_key)]] = 1
                            
                            common_doc_file = os.path.join(root, str(common_doc)+'.json')
                            common_doc_file = json.load(open(common_doc_file, 'r'))
                            common_tag_users = common_doc_file['users']
                            for user in users: 
                                for common_tag_user in common_tag_users:
                                    adj_matrix[user2id[str(common_tag_user)], user2id[str(user)]] = 1
                                    adj_matrix[user2id[str(user)], user2id[str(common_tag_user)]] = 1
                            
                            
                            # for context in user_contexts:
                            #     common_doc_users_file = os.path.join(root, str(common_doc)+'.json')
                            #     common_doc_users = json.load(open(common_doc_users_file, 'r'))
                            #     users = common_doc_users['users']
                            #     for user in users:
                            #         additional_users_file = './data/complete_data/pheme/'+context+str(user)+'.json'
                            #         add_users = json.load(open(additional_users_file, 'r'))
                            #         followers = add_users['followers'] if context == 'user_followers_filtered' else add_users['following']   
                            #         followers = list(map(int, followers))
                            #         for follower in followers:
                            #             if follower in all_users:
                            #                 adj_matrix[doc2id[str(doc_key)], user2id[str(follower)]]=1
                            #                 adj_matrix[user2id[str(follower)], doc2id[str(doc_key)]]=1
                                
                        if count%print_iter==0:
                            print("{} / {} done..".format(count+1, num_docs))
            
            else:
                print("\nPreparing entries for doc-user pairs...")
                src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
                train_docs = json.load(open('./data/complete_data/{}/train_docs.json'.format(dataset), 'r'))
                train_docs = train_docs['train_docs']
                user_splits = json.load(open('./data/complete_data/{}/user_splits.json'.format(dataset), 'r'))
                train_users = user_splits['train_users']
                not_found=0
                not_in_train, in_both, only_train = 0,0,0
                for root, dirs, files in os.walk(src_dir):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        doc_key = file.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        users = src_file['users'][:self.subset]
                        if str(doc_key) in train_docs:
                            for user in users:  
                                if str(doc_key) in doc2id and str(user) in user2id:
                                    only_train+=1
                                    adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                                    adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                                else:
                                    not_found+=1
                        else:
                            for user in users:  
                                if (str(doc_key) in doc2id) and (str(user) in user2id) and (str(user) not in train_users):
                                    not_in_train+=1
                                    adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                                    adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                                else:
                                    in_both+=1
                            


            end = time.time() 
            hrs, mins, secs = self.calc_elapsed_time(start, end)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))  
            print("Not Found users = ", not_found)
            print("Not in train users (only in val and test) = ", not_in_train)
            print("Users in both (train and val/test) = ", in_both)
            print("Non-zero entries = ", adj_matrix.getnnz())
            # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            # Creating the adjacency matrix (user-user edges)
            user_contexts = ['user_followers_filtered', 'user_following_filtered']
            start = time.time()
            key_errors, not_found, overlaps = 0,0,0
            print("\nPreparing entries for user-user pairs...")
            print_iter = int(num_users/10)
            print("Printing every {}  users done".format(print_iter))
            
            for user_context in user_contexts:
                print("    - from {}  folder...".format(user_context))
                src_dir2 = os.path.join(self.data_dir, 'complete_data', dataset, user_context)
                for root, dirs, files in os.walk(src_dir2):
                    for count, file in enumerate(files):
                        src_file_path = os.path.join(root, file)
                        # user_id = src_file_path.split(".")[0]
                        src_file = json.load(open(src_file_path, 'r'))
                        user_id = int(src_file['user_id'])
                        if str(user_id) in user2id:
                            followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file['following']   
                            followers = list(map(int, followers))
                            for follower in followers:
                                if str(follower) in user2id:
                                    adj_matrix[user2id[str(user_id)], user2id[str(follower)]]=1
                                    adj_matrix[user2id[str(follower)], user2id[str(user_id)]]=1
                                    
                        else:
                            not_found +=1
                        if count%print_iter==0:
                            # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                            print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, len(np.nonzero(adj_matrix)[0])))
                                         
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            # print("Key errors = ", key_errors)
            # print("Overlaps = ", overlaps)
            print("Not found user_ids = ", not_found)
            print("Total Non-zero entries = ", adj_matrix.getnnz())
            # print("Total Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))
            
            filename = self.data_dir+ '/complete_data' + '/adj_matrix_excl_{}.npz'.format(dataset)
            # filename = self.data_dir+ '/complete_data' + '/adj_matrix_{}.npy'.format(dataset)
            print("\nMatrix construction done! Saving in  {}".format(filename))
            save_npz(filename, adj_matrix.tocsr())
            # np.save(filename, adj_matrix)
            
            # Creating an edge_list matrix of adj_matrix as required by some GCN frameworks
            print("\nCreating edge_index format of adj_matrix...")
            start = time.time()
            # G = nx.DiGraph(adj_matrix.tocsr())
            # temp_matrix = adj_matrix.toarray()
            # rows, cols = np.nonzero(temp_matrix)
            rows, cols = adj_matrix.nonzero()
            
            edge_index = np.vstack((np.array(rows), np.array(cols)))
            print("Edge index shape = ", edge_index.shape)
            
            edge_matrix_file = self.data_dir+ '/complete_data' + '/adj_matrix_excl_{}_edge.npy'.format(dataset)
            print("saving edge_list format in :  ", edge_matrix_file)
            np.save(edge_matrix_file, edge_index, allow_pickle=True)
            
            # nx.write_edgelist(G, edge_matrix_file)
            # num_edges = len(G.edges())
            # print("No. of edges =  ", num_edges)           
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
    
    
    
    
    
    def create_feat_matrix(self, binary=True):
        labels = ['fake', 'real']
        for dataset in self.datasets:
            # print("\nPreparing feat_matrix for  {}: \n".format(dataset) + '-'*40)
            print("\n\n" + "-"*100 + "\n \t\t   Analyzing  {} dataset for feature_matrix\n".format(dataset) + '-'*100)
            doc2id_file = os.path.join(self.data_dir, 'complete_data', 'doc2id_'+dataset)
            test_docs_file = os.path.join(self.data_dir, 'complete_data', dataset, 'test_docs')
            train_docs_file = os.path.join(self.data_dir, 'complete_data', dataset, 'train_docs')
            user2id_file = os.path.join(self.data_dir, 'complete_data', 'user2id_'+dataset)
            
            doc2id = json.load(open(doc2id_file+'.json', 'r'))
            test_docs = json.load(open(test_docs_file+'.json', 'r'))
            train_docs = json.load(open(train_docs_file+'.json', 'r'))
            test_docs = test_docs['test_docs']
            train_docs= train_docs['train_docs']
            user2id = json.load(open(user2id_file+'.json', 'r'))
            N = len(doc2id) + len(user2id)
            
            vocab = {}
            vocab_size=0
            start = time.time()
            print("\nBuilding vocabulary...") 
            if dataset != 'pheme':               
                for label in labels:
                    src_doc_dir = os.path.join(self.data_dir, 'base_data', dataset, label)
                    for root, dirs, files in os.walk(src_doc_dir):
                        for file in files:
                            doc = file.split('.')[0]
                            if str(doc) in train_docs:
                                src_file_path = os.path.join(root, file)
                                with open(src_file_path, 'r') as f:
                                    file_content = json.load(f)
                                    text = file_content['text'].lower()[:500]
                                    text = re.sub(r'#[\w-]+', 'hashtag', text)
                                    text = re.sub(r'https?://\S+', 'url', text)
                                    # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                                    text = nltk.word_tokenize(text)
                                    for token in text:
                                        if token not in vocab.keys():
                                            vocab[token] = vocab_size
                                            vocab_size+=1
            else:
                vocab_size=0
                c=0
                src_doc_dir = os.path.join('./data/base_data/', 'pheme_cv')
                # src_doc_dir = os.path.join(self.data_dir, 'base_data', dataset, label)
                for root, dirs, files in os.walk(src_doc_dir):
                    for file in files:
                        if file.startswith('.') or file.startswith('structure') or file.startswith('annotation') or root.endswith('reactions'):
                            continue
                        else:
                            src_file_path = os.path.join(root, file)
                            src_tweet = json.load(open(src_file_path, 'r'))
                            text = src_tweet['text'].lower()
                            text = re.sub(r'#[\w-]+', 'hashtag', text)
                            text = re.sub(r'https?://\S+', 'url', text)
                            text = text.replace('\n', ' ')
                            text = text.replace('\t', ' ')
                            # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                            text = nltk.word_tokenize(text)
                            for token in text:
                                if token not in vocab.keys():
                                    vocab[token] = vocab_size
                                    vocab_size+=1  

            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            print("Size of vocab =  ", vocab_size)
            vocab_file = os.path.join(self.data_dir, 'complete_data', 'vocab_'+dataset)
            print("Saving vocab for  {}  at:  {}".format(dataset, vocab_file))
            with open(vocab_file+'.json', 'w+') as v:
                json.dump(vocab, v)
            
            
            feat_matrix = lil_matrix((N,vocab_size))
            print("\nSize of feature matrix = ", feat_matrix.shape)
            print("\nCreating feat_matrix entries for docs nodes...")
            start = time.time()
            if dataset != 'pheme':
                for label in labels:
                    src_doc_dir = os.path.join(self.data_dir, 'base_data', dataset, label)
                    for root, dirs, files in os.walk(src_doc_dir):
                        for count, file in enumerate(files):
                            print_iter = int(len(files) / 5)
                            doc_name = file.split('.')[0]
                            file = os.path.join(root, file)
                            # if str(doc_name) in doc2id:
                            #     feat_matrix[doc2id[str(doc_name)], :] = np.random.random(len(vocab)) > 0.9875
                            with open(file, 'r') as f:
                                file_content = json.load(f)
                                text = file_content['text'].lower()[:500]
                                text = re.sub(r'#[\w-]+', 'hashtag', text)
                                text = re.sub(r'https?://\S+', 'url', text)
                                text = text.replace('\t', ' ')
                                text = text.replace('\n', ' ')
                                # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                                text = nltk.word_tokenize(text)
                                for token in text:
                                    if token in vocab.keys():
                                        if str(doc_name) in doc2id:
                                            feat_matrix[doc2id[str(doc_name)], vocab[token]] = 1 
                            if count%print_iter==0:
                                print("{} / {} done..".format(count+1, len(files)))
            else:
                c=0
                src_doc_dir = os.path.join('./data/base_data/', 'pheme_cv')
                # src_doc_dir = os.path.join(self.data_dir, 'base_data', dataset, label)
                for root, dirs, files in os.walk(src_doc_dir):
                    for file in files:
                        if file.startswith('.') or file.startswith('structure') or file.startswith('annotation') or root.endswith('reactions'):
                            continue
                        else:
                            src_file_path = os.path.join(root, file)
                            src_tweet = json.load(open(src_file_path, 'r'))
                            doc = root.split('\\')[-2]
                            text = src_tweet['text'].lower()
                            text = re.sub(r'#[\w-]+', 'hashtag', text)
                            text = re.sub(r'https?://\S+', 'url', text)
                            text = text.replace('\n', ' ')
                            text = text.replace('\t', ' ')
                            # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                            text = nltk.word_tokenize(text)
                            for token in text:
                                if token in vocab.keys():
                                    feat_matrix[doc2id[str(doc)], vocab[token]] = 1
                            c+=1
                            if c%500 == 0:
                                print("{} done...".format(c))
                
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            
            sum_1 = np.array(feat_matrix.sum(axis=1)).squeeze(1)
            print(sum_1.shape)
            idx = np.where(sum_1==0)
            print(len(idx[0]))
            
            
            print("\nCreating feat_matrix entries for users nodes...")
            start = time.time()
            not_found, use = 0,0
            user_splits = json.load(open('./data/complete_data/{}/user_splits.json'.format(dataset), 'r'))
            train_users = user_splits['train_users']
            src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
            user_contexts = ['user_followers_filtered', 'user_following_filtered']
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    print_iter = int(len(files) / 10)
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open (src_file_path, 'r'))
                    users = src_file['users'][:self.subset]
                    doc_key = file.split(".")[0]
                    # if str(doc_key) in train_docs:
                    # Each user of this doc has its features as the features of the doc
                    for user in users:
                        use+=1
                        if (str(doc_key) in train_docs):
                            if str(doc_key) in doc2id and str(user) in user2id:
                                feat_matrix[user2id[str(user)], :] += feat_matrix[doc2id[str(doc_key)], :]
                                # Each follow-er/ing user of this user has the feature matrix of this user
                                # for user_context in user_contexts:
                                #     src_dir2 = os.path.join(self.data_dir, 'complete_data', dataset, user_context)
                                #     src_file_path = os.path.join(src_dir2, str(user)+'.json')
                                #     # user_id = src_file_path.split(".")[0]
                                #     if os.path.isfile(src_file_path):
                                #         src_file = json.load(open(src_file_path, 'r'))
                                #         followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file['following']
                                #         followers = list(map(int, followers))
                                #         for follower in followers:
                                #             if follower in user2id.keys():
                                #                 feat_matrix[user2id[str(follower)], :] += feat_matrix[user2id[str(user)], :]
                                #             else:
                                #                 not_found+=1
                            
                        elif str(doc_key) not in train_docs and str(user) not in train_users:
                            if str(doc_key) in doc2id and str(user) in user2id:
                                feat_matrix[user2id[str(user)], :] += feat_matrix[doc2id[str(doc_key)], :]
                                # Each follow-er/ing user of this user has the feature matrix of this user
                                # for user_context in user_contexts:
                                #     src_dir2 = os.path.join(self.data_dir, 'complete_data', dataset, user_context)
                                #     src_file_path = os.path.join(src_dir2, str(user)+'.json')
                                #     # user_id = src_file_path.split(".")[0]
                                #     if os.path.isfile(src_file_path):
                                #         src_file = json.load(open(src_file_path, 'r'))
                                #         followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file['following']
                                #         followers = list(map(int, followers))
                                #         for follower in followers:
                                #             if follower in user2id.keys():
                                #                 feat_matrix[user2id[str(follower)], :] += feat_matrix[user2id[str(user)], :]
                                #             else:
                                #                 not_found+=1
                    if count%print_iter==0:
                        print(" {} / {} done..".format(count+1, len(files)))
                    
            hrs, mins, secs = self.calc_elapsed_time(start, time.time())
            print(not_found, use)
            print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
            
            feat_matrix = feat_matrix >= 1
            feat_matrix = feat_matrix.astype(int)
            
            # Sanity Checks
            sum_1 = np.array(feat_matrix.sum(axis=1)).squeeze(1)
            print(sum_1.shape)
            idx = np.where(sum_1==0)
            print(len(idx[0]))
            print(idx)
            
            filename = os.path.join(self.data_dir, 'complete_data')
            filename = filename+'/feat_matrix_{}.npz'.format(dataset)
            print("Matrix construction done! Saving in :   {}".format(filename))
            save_npz(filename, feat_matrix.tocsr())
            
            
            
    def convert_annotations(self, annotation, string = False):
        if 'misinformation' in annotation.keys() and 'true'in annotation.keys():
            if int(annotation['misinformation'])==0 and int(annotation['true'])==0:
                if string:
                    label = "unverified"
                else:
                    label = 2
            elif int(annotation['misinformation'])==0 and int(annotation['true'])==1 :
                if string:
                    label = "true"
                else:
                    label = 1
            elif int(annotation['misinformation'])==1 and int(annotation['true'])==0 :
                if string:
                    label = "false"
                else:
                    label = 0
            elif int(annotation['misinformation'])==1 and int(annotation['true'])==1:
                print ("OMG! They both are 1!")
                print(annotation['misinformation'])
                print(annotation['true'])
                label = None
                
        elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
            # all instances have misinfo label but don't have true label
            if int(annotation['misinformation'])==0:
                if string:
                    label = "unverified"
                else:
                    label = 2
            elif int(annotation['misinformation'])==1:
                if string:
                    label = "false"
                else:
                    label = 0
                    
        elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
            print ('Has true not misinformation')
            label = None
        else:
            print('No annotations')
            label = None              
        return label
      
    """
    Create labels for each node of the graph
    """
    def create_labels(self):
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t   Analyzing  {} dataset for Creating Labels\n".format(dataset) + '-'*100)
            doc2id_file = os.path.join(self.data_dir, 'complete_data', 'doc2id_'+dataset)
            adj_matrix_file = os.path.join(self.data_dir, 'complete_data', 'adj_matrix_'+dataset)
            
            doc2id = json.load(open(doc2id_file+'.json', 'r')) 
            adj_matrix = load_npz(adj_matrix_file+'.npz')
            N,_ = adj_matrix.shape
            del adj_matrix
            print("\nCreating doc2labels dictionary...")
            doc2labels = {}
            c=0
            if dataset== 'pheme':
                pheme_dir = './data/base_data/pheme_cv'
                for root, dirs, files in os.walk(pheme_dir):
                    for file in files:
                        if not file.startswith('annotation'):
                            continue
                        else:
                            src_file_path = os.path.join(root, file)
                            doc = root.split('\\')[-1]
                            with open(src_file_path, 'r') as j:
                                annotation = json.load(j)
                                doc2labels[str(doc)] = self.convert_annotations(annotation, string = False)
                                c+=1
                                if c%500 == 0:
                                    print("{} done..".format(c))
            
            else:
                user_contexts = ['fake', 'real']
                for user_context in user_contexts:
                    data_dir = os.path.join(self.data_dir, 'complete_data', dataset, user_context)
                    label = 1 if user_context=='fake' else 0
                    for root, dirs, files in os.walk(data_dir):
                        for count,file in enumerate(files):
                            doc= root.split('\\')[-1]
                            doc2labels[str(doc)] = label
            
            assert len(doc2labels.keys()) == len(doc2id.keys())
            labels_dict_file = os.path.join(self.data_dir, 'complete_data')
            labels_dict_file = labels_dict_file+'/doc2labels_'+dataset+'.json'
            print("Saving labels_dict for  {}  at:  {}".format(dataset, labels_dict_file))
            with open(labels_dict_file, 'w+') as v:
                json.dump(doc2labels, v)
            
            labels_list = np.zeros(N)
            for key,value in doc2labels.items():
                labels_list[doc2id[str(key)]] = value
                          
            # Sanity Checks
            # print(sum(labels_list))
            # print(len(labels_list))
            # print(sum(labels_list[2402:]))
            # print(sum(labels_list[:2402]))
            
            filename = os.path.join(self.data_dir, 'complete_data')
            filename = filename+'/labels_list_{}.json'.format(dataset)
            temp_dict = {}
            temp_dict['labels_list'] = list(labels_list)
            print("Labels list construction done! Saving in :   {}".format(filename))
            with open(filename, 'w+') as v:
                json.dump(temp_dict, v)
            
            
            # Create the all_labels file
            all_labels = np.zeros(N)
            all_labels_data_file = './data/complete_data/all_labels_{}.json'.format(dataset)
            for doc in doc2labels.keys():
                all_labels[doc2id[str(doc)]] = doc2labels[str(doc)]
            
            temp_dict = {}
            temp_dict['all_labels'] = list(all_labels)
            print("Sum of labels this test set = ", sum(all_labels))
            print("Len of labels = ", len(all_labels))
            with open(all_labels_data_file, 'w+') as j:
                json.dump(temp_dict, j)
        return None
    
    
    def create_split_masks(self):
        print("\n\n" + "-"*100 + "\n \t\t   Creating data-split masks for PHEME\n" + '-'*100)
        with open('./data/complete_data/doc2labels_pheme.json','r') as j:
            doc2labels = json.load(j)
        
        adj_matrix_file = os.path.join(self.data_dir, 'complete_data', 'adj_matrix_pheme')
        doc2id_file = os.path.join(self.data_dir, 'complete_data', 'doc2id_pheme')
        
        adj_matrix = load_npz(adj_matrix_file+'.npz')
        doc2id = json.load(open(doc2id_file+'.json', 'r'))
        N,_ = adj_matrix.shape
        del adj_matrix
        
        
        # Creating the CV folds from the remaining events
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        src_dir = os.path.join(self.data_dir, 'base_data', 'pheme_cv')
        
        for fold, event in enumerate(events):
            print("\nCreating fold_{}  with  {}  as test set\n".format(fold+1, event) + "-"*50 )
            train_mask, test_mask, train_labels_mask, test_labels_mask = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
            representation_mask = np.ones(N)
            test_event = event
            train_events = events.copy()
            train_events.remove(event)
            
            print("Test set: \n" + "-"*20)
            test_data_dir = os.path.join(src_dir, test_event)
            test_labels_file = './data/complete_data/pheme_cv/test_labels_{}.json'.format(fold+1)
            test_data_file = './data/complete_data/pheme_cv/test_mask_{}.json'.format(fold+1)
            repr_file = './data/complete_data/pheme_cv/repr_mask_{}.json'.format(fold+1)
            c=0
            for root, dirs, files in os.walk(test_data_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or root.endswith('reactions') or file.startswith('annotation'):
                        continue
                    else:
                        doc = file.split(".")[0]
                        test_labels_mask[doc2id[str(doc)]] = doc2labels[str(doc)]
                        test_mask[doc2id[str(doc)]] = 1
                        representation_mask[doc2id[str(doc)]] = 0
                        c+=1
                        if c%200 == 0:
                            print("{} done...".format(c))
            temp_dict = {}
            temp_dict['test_mask'] = list(test_mask)
            print("Sum of this test set = ", sum(test_mask))
            print("Len of this mask = ", len(test_mask))
            with open(test_data_file, 'w+') as j:
                json.dump(temp_dict, j)
            
            temp_dict = {}
            temp_dict['test_labels_mask'] = list(test_labels_mask)
            print("Sum of labels this test set = ", sum(test_labels_mask))
            with open(test_labels_file, 'w+') as j:
                json.dump(temp_dict, j)
                
            temp_dict = {}
            temp_dict['repr_mask'] = list(representation_mask)
            print("Sum of representation mask = ", sum(representation_mask))
            with open(repr_file, 'w+') as j:
                json.dump(temp_dict, j)
                
            print("\nTrain set: \n" + "-"*20)
            train_labels_file = './data/complete_data/pheme_cv/train_labels_{}.json'.format(fold+1)
            train_data_file = './data/complete_data/pheme_cv/train_mask_{}.json'.format(fold+1)
            c=0
            for train_event in train_events:
                train_data_dir = os.path.join(src_dir, train_event)
                for root, dirs, files in os.walk(train_data_dir):
                    for file in files:
                        if file.startswith('.') or file.startswith('structure') or root.endswith('reactions') or file.startswith('annotation'):
                            continue
                        else:
                            doc = file.split(".")[0]
                            train_labels_mask[doc2id[str(doc)]] = doc2labels[str(doc)]
                            train_mask[doc2id[str(doc)]] = 1
                            c+=1
                            if c%200 == 0:
                                print("{} done...".format(c))
            temp_dict = {}
            temp_dict['train_mask'] = list(train_mask)
            print("Sum of this train set = ", sum(train_mask))
            with open(train_data_file, 'w+') as j:
                json.dump(temp_dict, j)
                        
            temp_dict = {}
            temp_dict['train_labels_mask'] = list(train_labels_mask)
            print("Sum of labels of this train set = ", sum(train_labels_mask))
            with open(train_labels_file, 'w+') as j:
                json.dump(temp_dict, j)
        return None
    
    
    def create_user_splits_list(self):
        # Create and save users present in the splits
        for dataset in self.datasets:
            test_docs = json.load(open('./data/complete_data/{}/test_docs.json'.format(dataset), 'r'))
            train_docs = json.load(open('./data/complete_data/{}/train_docs.json'.format(dataset), 'r'))
            val_docs = json.load(open('./data/complete_data/{}/val_docs.json'.format(dataset), 'r'))
            
            train_docs = train_docs['train_docs']
            val_docs = val_docs['val_docs']
            test_docs = test_docs['test_docs']
            
            print("\nCreating users in splits file..")
            src_dir = os.path.join(self.data_dir, 'complete_data', dataset, 'complete')
            train_users, val_users, test_users = set(), set(), set()
            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    doc_key = file.split('.')[0]
                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open(src_file_path, 'r'))
                    users = src_file['users'][:self.subset]
                    users = [str(s) for s in users if isinstance(s, int)]
                    if str(doc_key) in train_docs:
                        train_users.update(users)
                    if str(doc_key) in val_docs:
                        val_users.update(users)
                    if str(doc_key) in test_docs:
                        test_users.update(users)
            
            temp_dict = {}
            temp_dict['train_users'] = list(train_users)
            temp_dict['val_users'] = list(val_users)
            temp_dict['test_users'] = list(test_users)
            with open('./data/complete_data/{}/user_splits.json'.format(dataset), 'w+') as j:
                json.dump(temp_dict, j)
                
        return None
    
    
    
    def create_random_split_masks(self):
        
        for dataset in self.datasets:
            print("\n\n" + "-"*100 + "\n \t\t   Creating data-split masks for {}\n".format(dataset) + '-'*100)
            
            with open('./data/complete_data/doc2labels_{}.json'.format(dataset),'r') as j:
               doc2labels = json.load(j)
            
            adj_matrix_file = os.path.join(self.data_dir, 'complete_data', 'adj_matrix_{}'.format(dataset))
            doc2id_file = os.path.join(self.data_dir, 'complete_data', 'doc2id_{}'.format(dataset))
            
            adj_matrix = load_npz(adj_matrix_file+'.npz')
            doc2id = json.load(open(doc2id_file+'.json', 'r'))
            N,_ = adj_matrix.shape
            del adj_matrix
            
            train_mask, test_mask, val_mask = np.zeros(N), np.zeros(N), np.zeros(N)
            representation_mask = np.ones(N)
            # representation_mask = np.zeros(N)
            print("Len of doc2id = ", len(doc2id))
            
            test_data_file = './data/complete_data/{}/test_mask_rand.json'.format(dataset)
            val_data_file = './data/complete_data/{}/val_mask_rand.json'.format(dataset)
            train_data_file = './data/complete_data/{}/train_mask_rand.json'.format(dataset)
            repr_file = './data/complete_data/{}/repr_mask_rand.json'.format(dataset)
            
            
            if dataset == 'pheme':
                pos, neg, unverif = [], [], []
                repr_file = './data/complete_data/pheme_cv/repr_mask_rand.json'
                
                for doc, label in doc2labels.items():
                    if label==0:
                        neg.append(doc2id[str(doc)])
                    elif label==1:
                        pos.append(doc2id[str(doc)])
                    elif label==2:
                        unverif.append(doc2id[str(doc)])
                    else:
                        raise ValueError("[!] Error: Label should be 0,1,2 but found ", label)
                    
                print("No. of pos, neg, and unverif labels are : {}, {}, {}".format(len(pos), len(neg), len(unverif)))
                
                # pos_idx = list(range(len(pos)))
                # neg_idx = list(range(len(neg)))
                # unverif_idx = list(range(len(unverif)))    
                pos_idx = pos
                neg_idx = neg
                unverif_idx = unverif
        
                random.shuffle(pos_idx)
                random.shuffle(neg_idx)
                random.shuffle(unverif_idx)
                        
                test_pos_size = int(np.ceil(0.2 * len(pos)))
                train_pos_size = int(len(pos) - test_pos_size)
                test_neg_size = int(np.ceil(0.2 * len(neg)))
                train_neg_size = int(len(neg) - test_neg_size)
                test_unverif_size = int(np.ceil(0.2 * len(unverif)))
                train_unverif_size = int(len(unverif) - test_unverif_size)
                
                train_idx = pos_idx[ :train_pos_size] + neg_idx[ :train_neg_size] + unverif_idx[ :train_unverif_size]
                test_idx = pos_idx[train_pos_size: ] + neg_idx[train_neg_size: ] + unverif_idx[train_unverif_size: ]
                            
                not_in_either=0
                train_labels, test_labels = [], []
                for doc, id in doc2id.items():
                    if id in train_idx:
                        # print("train_idx = ", id)
                        train_mask[id] = 1
                        train_labels.append(doc2labels[str(doc)])
                    elif id in test_idx:
                        # print("test_idx = ", id)
                        test_mask[id] = 1
                        test_labels.append(doc2labels[str(doc)])
                        representation_mask[doc2id[str(doc)]] = 0
                    else:
                        # print("not found = ", id)
                        not_in_either+=1
                
                print("\nnot_in_either = ", not_in_either)
                print("train_mask sum = ", sum(train_mask))
                print("test_mask sum = ", sum(test_mask))
                
                true, false, unverif = self.get_label_distribution_pheme(train_labels)
                print("\nTrue labels in train = {:.2f} %".format(true*100))
                print("False labels in train = {:.2f} %".format(false*100))
                print("Unverified labels in train = {:.2f} %".format(unverif*100))
                
                
                true, false, unverif = self.get_label_distribution_pheme(test_labels)
                print("\nTrue labels in test = {:.2f} %".format(true*100))
                print("False labels in test = {:.2f} %".format(false*100))
                print("Unverified labels in test = {:.2f} %".format(unverif*100))
                
                temp_dict = {}
                temp_dict['test_mask'] = list(test_mask)
                print("Len of this mask = ", len(test_mask))
                with open(test_data_file, 'w+') as j:
                    json.dump(temp_dict, j)
                
                temp_dict = {}
                temp_dict['train_mask'] = list(train_mask)
                print("Len of this train set = ", len(train_mask))
                with open(train_data_file, 'w+') as j:
                    json.dump(temp_dict, j)
                
                temp_dict = {}
                temp_dict['repr_mask'] = list(representation_mask)
                print("Sum of representation mask = ", sum(representation_mask))
                with open(repr_file, 'w+') as j:
                    json.dump(temp_dict, j)
                    
                    
            else:               
                fake, real = [], []
                                
                for doc, label in doc2labels.items():
                    if label==0:
                        real.append(doc2id[str(doc)])
                    elif label==1:
                        fake.append(doc2id[str(doc)])
                    else:
                        raise ValueError("[!] Error: Label should be 0,1 but found ", label)
                    
                print("No. of fake and real labels are : {}, {}".format(len(fake), len(real)))
                
                # pos_idx = list(range(len(pos)))
                # neg_idx = list(range(len(neg)))
                # unverif_idx = list(range(len(unverif)))    
                pos_idx = fake
                neg_idx = real
        
                random.shuffle(pos_idx)
                random.shuffle(neg_idx)        
                
                test_pos_size = int(np.ceil(0.2 * len(fake)))
                val_pos_size = int(np.ceil(0.1 * len(fake)))
                train_pos_size = int(len(fake) - test_pos_size - val_pos_size)
                test_neg_size = int(np.ceil(0.2 * len(real)))
                val_neg_size = int(np.ceil(0.1 * len(real)))
                train_neg_size = int(len(real) - test_neg_size - val_neg_size)
                
                train_idx = pos_idx[ :train_pos_size] + neg_idx[ :train_neg_size] 
                val_idx = pos_idx[train_pos_size : train_pos_size+val_pos_size] + neg_idx[train_neg_size : train_neg_size+val_neg_size] 
                test_idx = pos_idx[train_pos_size+val_pos_size: ] + neg_idx[train_neg_size+val_neg_size: ] 
                            
                not_in_either=0
                train_docs, val_docs, test_docs = [], [], []
                train_labels, val_labels, test_labels = [], [], []
                for doc, id in doc2id.items():
                    if id in train_idx:
                        train_docs.append(doc)
                        train_mask[id] = 1
                        train_labels.append(doc2labels[str(doc)])
                        # representation_mask[doc2id[str(doc)]] = 0
                    elif id in val_idx:
                        val_docs.append(doc)
                        val_mask[id] = 1
                        val_labels.append(doc2labels[str(doc)])
                        # representation_mask[doc2id[str(doc)]] = 0
                    elif id in test_idx:
                        test_docs.append(doc)
                        test_mask[id] = 1
                        test_labels.append(doc2labels[str(doc)])
                        representation_mask[doc2id[str(doc)]] = 0
                    else:
                        not_in_either+=1
                
                print("\nNot_in_either = ", not_in_either)
                print("train_mask sum = ", sum(train_mask))
                print("val_mask sum = ", sum(val_mask))
                print("test_mask sum = ", sum(test_mask))
                
                fake, real = self.get_label_distribution(train_labels)
                print("\nFake labels in train = {:.2f} %".format(fake*100))
                print("Real labels in train = {:.2f} %".format(real*100))
                
                fake, real = self.get_label_distribution(val_labels)
                print("\nFake labels in val = {:.2f} %".format(fake*100))
                print("Real labels in val = {:.2f} %".format(real*100))
                
                fake, real = self.get_label_distribution(test_labels)
                print("\nFake labels in test = {:.2f} %".format(fake*100))
                print("Real labels in test = {:.2f} %".format(real*100))
                
                temp_dict = {}
                temp_dict['test_mask'] = list(test_mask)
                with open(test_data_file, 'w+') as j:
                    json.dump(temp_dict, j)
                
                temp_dict = {}
                temp_dict['test_docs'] = test_docs
                with open('./data/complete_data/{}/test_docs.json'.format(dataset), 'w+') as j:
                    json.dump(temp_dict, j)
                    
                temp_dict = {}
                temp_dict['val_mask'] = list(val_mask)
                with open(val_data_file, 'w+') as j:
                    json.dump(temp_dict, j)
                
                temp_dict = {}
                temp_dict['val_docs'] = val_docs
                with open('./data/complete_data/{}/val_docs.json'.format(dataset), 'w+') as j:
                    json.dump(temp_dict, j)
                
                temp_dict = {}
                temp_dict['train_mask'] = list(train_mask)
                with open(train_data_file, 'w+') as j:
                    json.dump(temp_dict, j)
                
                temp_dict = {}
                temp_dict['train_docs'] = train_docs
                with open('./data/complete_data/{}/train_docs.json'.format(dataset), 'w+') as j:
                    json.dump(temp_dict, j)
                    
                temp_dict = {}
                temp_dict['repr_mask'] = list(representation_mask)
                print("Sum of representation mask = ", sum(representation_mask))
                with open(repr_file, 'w+') as j:
                    json.dump(temp_dict, j)               
                  
        return None
    
    

    def get_label_distribution(self, labels):  
        fake = labels.count(1)
        real = labels.count(0)
        denom = fake+real
        return fake/denom, real/denom
    
    
    def get_label_distribution_pheme(self, labels):  
        true = labels.count(1)
        false = labels.count(0)
        unverified = labels.count(2)
        denom = true + false + unverified    
        return true/denom, false/denom, unverified/denom

    def calc_elapsed_time(self, start, end):
        hours, rem = divmod(end-start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), int(seconds)
    
            
            
    def generate_graph_stats(self):
        print("\n\n" + "-"*100 + "\n \t\t   Checking overlapping users for PHEME\n" + '-'*100)  
       
        # Creating the CV folds from the remaining events
        events = ['ch', 'ebola', 'ferg', 'german', 'gurlitt', 'ottawa', 'putin', 'sydney', 'toronto']
        src_dir = os.path.join(self.data_dir, 'base_data', 'pheme_cv')
        users =  defaultdict(set)
        user_contexts = ['user_followers_filtered', 'user_following_filtered']
        for fold, event in enumerate(events):
            print("\nCreating fold_{}  with  {}  as test set\n".format(fold+1, event) + "-"*50 )
            test_data_dir = os.path.join(src_dir, event)
            c=0
            for root, dirs, files in os.walk(test_data_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('structure') or file.startswith('annotation'):
                        continue
                    else:
                        src_file_path = os.path.join(root, file)
                        src_file = json.load(open(src_file_path, 'r'))
                        user_id = int(src_file['user']['id'])
                        users[event].update([user_id])
                        for user_context in user_contexts:
                            src_file_path = os.path.join(self.data_dir, 'complete_data', 'pheme', user_context, str(user_id)+'.json')
                            # user_id = src_file_path.split(".")[0]
                            src_file = json.load(open(src_file_path, 'r'))
                            followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file['following']   
                            followers = list(map(int, followers))
                            for follower in followers:
                                users[event].update([follower])
                        c+=1
                        if c%2000 == 0:
                            print("{} done...".format(c))
            print("Total = ", c)
        all_users = set()
        for event, user in users.items():
            print("No. of users in {} = {}".format(event, len(user)))
            all_users.update(user)
        all_users = list(all_users)
        
    
        
        # # Fold-wise heatmap
        # heatmap = np.zeros((len(events),2))
        # for i, event in enumerate(events):
        #     event_users = users[event]
        #     rest_users = set()
        #     rest_events = events.copy()
        #     rest_events.remove(event)
            
        #     for rest in rest_events:
        #         rest_users.update(users[rest])
            
        #     common = rest_users.intersection(event_users)
        #     heatmap[i,0] = len(rest_users)/len(rest_users)
        #     heatmap[i,1] = len(common)/len(rest_users)
        #     # total = event_users.union(rest_users)
        
        # # fig2= plt.figure(2)
        # # Plot the degrees as Heatmap
        # bx = sns.heatmap(heatmap, linewidth=0.5, cmap="YlGnBu", annot=True, annot_kws={"size": 7})
        # bx.set_xticklabels(['train', 'test'])
        # bx.set_yticklabels(events, rotation=0)
        # plt.show()
        
        
        # Event-wise heatmap (avg degree in/out)
        heatmap = np.zeros((len(events), len(events)))
        norm = []
        for i in range(len(events)):
            for j in range(len(events)):
                a = set(users[events[i]])
                b = set(users[events[j]])
                if i==j:
                    norm.append(len(a.intersection(b)))
                print("Overlap of users between {} and {} =  {}".format(events[i], events[j], len(a.intersection(b))))
                heatmap[i,j] = len(a.intersection(b))/len(users[events[i]])
        

        # Plot the degrees as Heatmap
        print(heatmap.shape)
        ax = sns.heatmap(heatmap, linewidth=0.5, cmap="YlGnBu", annot=True, annot_kws={"size": 9})
        ax.set_xlabel("--> Avg out-degree")
        ax.set_ylabel("--> Avg in-degree")
        ax.set_xticklabels(events)
        ax.set_yticklabels(events, rotation=0)
        plt.show()
        
        return None
                
                
    
    def create_filtered_follower_following(self):
        for dataset in self.datasets:
            if dataset == 'pheme':
                with open('./data/complete_data/user2id_pheme.json','r') as j:
                   all_users = json.load(j)
                
                print("\n\n" + "-"*100 + "\n \t\t   Creating filtered follower-following\n" + '-'*100)
                user_contexts = ['user_followers', 'user_following']
                print_iter = int(len(all_users)/10)
                
                for user_context in user_contexts:
                    print("    - from {}  folder...".format(user_context))
                    src_dir2 = os.path.join(self.data_dir, 'complete_data', 'pheme', user_context)
                    dest_dir = os.path.join(self.data_dir, 'complete_data', 'pheme', user_context+'_filtered')
                    for root, dirs, files in os.walk(src_dir2):
                        for count, file in enumerate(files):
                            src_file_path = os.path.join(root, file)
                            # user_id = src_file_path.split(".")[0]
                            src_file = json.load(open(src_file_path, 'r'))
                            user_id = int(src_file['user_id'])
                            dest_file_path = os.path.join(dest_dir, str(user_id)+'.json')
                            if str(user_id) in all_users:
                                temp= set()
                                followers = src_file['followers'] if user_context == 'user_followers' else src_file['following']   
                                followers = list(map(int, followers))
                                for follower in followers:
                                    if str(follower) in all_users:
                                        temp.update([follower])
                                temp_dict = {}
                                temp_dict['user_id'] = user_id
                                name = 'followers' if user_context == 'user_followers' else 'following'
                                temp_dict[name] = list(temp)
                                with open(dest_file_path, 'w+') as v:
                                    json.dump(temp_dict, v)
                            else:
                                print("not found")
                            if count%print_iter==0:
                                # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                                print("{} done..".format(count+1))
                                
            else:
                with open('./data/complete_data/user2id_{}.json'.format(dataset),'r') as j:
                   all_users = json.load(j)
                
                done_users = json.load(open('./data/complete_data/{}/done_users_30.json'.format(dataset), 'r'))['done_users']
                print("Total done users = ", len(done_users))
                
                print("\n\n" + "-"*100 + "\n \t\t   Creating filtered follower-following\n" + '-'*100)
                user_contexts = ['user_followers', 'user_following']
                print_iter = int(len(all_users)/10)
                not_found=0
                
                for user_context in user_contexts:
                    print("    - from {}  folder...".format(user_context))
                    src_dir = os.path.join(self.data_dir, 'complete_data', dataset, user_context)
                    dest_dir = os.path.join(self.data_dir, 'complete_data', dataset, user_context+'_filtered')
                    for root, dirs, files in os.walk(src_dir):
                        for count, file in enumerate(files):
                            src_file_path = os.path.join(root, file)
                            # user_id = src_file_path.split(".")[0]
                            src_file = json.load(open(src_file_path, 'r'))
                            user_id = int(src_file['user_id'])
                            dest_file_path = os.path.join(dest_dir, str(user_id)+'.json')
                            if not os.path.isfile(dest_file_path):
                                if int(user_id) in done_users:
                                    temp= set()
                                    followers = src_file['followers'] if user_context == 'user_followers' else src_file['following']   
                                    followers = list(map(int, followers))
                                    for follower in followers:
                                        if int(follower) in done_users:
                                            temp.update([follower])
                                    temp_dict = {}
                                    temp_dict['user_id'] = user_id
                                    name = 'followers' if user_context == 'user_followers' else 'following'
                                    temp_dict[name] = list(temp)
                                    with open(dest_file_path, 'w+') as v:
                                        json.dump(temp_dict, v)
                                else:
                                    not_found+=1
                                    # print("{}  not found..".format(user_id))
                            if count%2000==0:
                                # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                                print("{} done..".format(count+1))
                print("\nNot found users = ", not_found)  
        return None
                
        
                
        
      


if __name__== '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = './data',
                          help='path to dataset folder that contains the folders to gossipcop or politifact folders (raw data)')
    parser.add_argument('--subset', type = int, default = 2500,
                          help='Create graph on a subset of unique users of each doc')
    
    parser.add_argument('--create_aggregate_folder', type = bool, default = False,
                          help='Aggregate only user ids from different folders of tweets/retweets to a single place')
    parser.add_argument('--create_adj_matrix', type = bool, default = False,
                          help='To create adjacency matrix for a given dataset')
    parser.add_argument('--create_feat_matrix', type = bool, default = False,
                          help='To create feature matrix for a given dataset')
    parser.add_argument('--create_labels', type = bool, default = True,
                          help='To create labels for all the nodes')
    parser.add_argument('--create_split_masks', type = bool, default = True,
                          help='To create node masks for data splits')
    
    
    args, unparsed = parser.parse_known_args()
    config = args.__dict__
    
    preprocesser = GCN_PreProcess(config, gossipcop=True, politifact = False, pheme=False)