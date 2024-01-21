import torch
import numpy as np
import pandas as pd
import os
from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm
import re
import torch
from icecream import ic

def extract_mentioned_usernames_in_tweet(tweet: str) -> list[str]:
    """
    Extracts mentioned usernames (Twitter handles) from a given tweet.

    Args:
        tweet (str): The tweet text from which to extract usernames.

    Returns:
        List[str]: A list of usernames mentioned in the tweet.
    """
    # Define a regular expression pattern to match Twitter handles
    pattern = r'@(\w+)'

    # Use the findall() method to find all matches in the tweet
    usernames = re.findall(pattern, tweet)

    return usernames


class Twibot20(Dataset):
    """
    Twibot20 is a Dataset class for handling the Twibot-20 dataset. It includes methods for 
    loading and processing data, extracting features, and creating a graph structure for 
    analysis purposes.
    """

    def __init__(self, root: str = './save_data/', device: str = 'cpu', process: bool = True, save: bool = True):
        """
        Initializes the Twibot20 dataset.

        Parameters:
        root (str): The root directory for the dataset files.
        device (str): The computing device ('cpu' or 'gpu') to be used for tensor operations.
        process (bool): Indicates whether to process the dataset upon initialization.
        save (bool): Specifies whether to save processed data.
        """
        self.root = root
        self.device = device
        self.process = process
        self.save = save

        if not os.path.exists(self.root):
            os.mkdir(self.root)

        if process:
            # Loading JSON data files for train, test, and dev datasets
            # print('Loading train.json')
            # df_train = pd.read_json(f'{root}Twibot-20/train.json')
            # df_train = df_train[df_train['domain'].apply(lambda x: x == ['Politics'])]

            # print('Loading test.json')
            # df_test = pd.read_json(f'{root}Twibot-20/test.json')
            # df_test = df_test[df_test['domain'].apply(lambda x: x == ['Politics'])]

            print('Loading dev.json')
            df_dev = pd.read_json(f'{root}dev.json')
            print(df_dev)
            print('Loading Finished')

            # Selecting relevant columns from each DataFrame
            columns_to_select = [0, 1, 2, 3, 5]
            # df_train = df_train.iloc[:, columns_to_select]
            # df_test = df_test.iloc[:, columns_to_select]
            df_dev = df_dev.iloc[:, columns_to_select]

            # Concatenating train, test, and dev DataFrames
            # self.df_data_labeled = pd.concat([df_train, df_dev, df_test], ignore_index=True)
            self.df_data_labeled = df_dev
            self.df_data = self.df_data_labeled.copy()
            print(f"Data length: {len(self.df_data)}")
        
    def load_labels(self) -> torch.Tensor:
        """
        Loads the labels for the dataset from a saved file, or creates them if they don't exist.

        Returns:
        torch.Tensor: A tensor containing the labels.
        """
        print('Loading labels...', end='   ')
        path = os.path.join(self.root, 'label.pt')

        if not os.path.exists(path):
            # Convert labels to a tensor and move it to the specified device
            labels = torch.LongTensor(self.df_data_labeled['label']).to(self.device)
            
            # Save the labels if the save flag is set
            if self.save:
                torch.save(labels, path)
        else:
            # Load the labels if they already exist
            labels = torch.load(path).to(self.device)

        print('Finished')
        return labels
    

    def Des_Preprocess(self) -> np.ndarray:
        """
        Preprocesses the 'description' feature of the dataset. It involves extracting the
        'description' field from the user profiles, handling missing values, and saving or loading
        the processed data as needed.

        Returns:
        np.ndarray: An array containing user profile descriptions.
        """
        print('Loading raw feature1...', end='   ')
        path = os.path.join(self.root, 'description.npy')

        if not os.path.exists(path):
            description = []

            # Iterate over each record in the dataset to extract the description
            for i in range(self.df_data.shape[0]):
                profile = self.df_data['profile'][i]

                # Handling missing or None values in the description
                if profile is None or profile.get('description') is None:
                    description.append('None')
                else:
                    description.append(profile['description'])

            # Convert the list to a NumPy array
            description = np.array(description)

            # Save the processed descriptions if the save flag is set
            if self.save:
                np.save(path, description)
        else:
            # Load the descriptions if they have already been processed and saved
            description = np.load(path, allow_pickle=True)

        print('Finished')
        return description

    def Des_embbeding(self) -> torch.Tensor:
        """
        Processes and embeds the description feature of the dataset using the RoBERTa model.
        If the embeddings are already saved, it loads them; otherwise, it calculates and saves them.

        Returns:
        torch.Tensor: A tensor containing the embedded features of the descriptions.
        """
        print('Running feature1 embedding')
        path = os.path.join(self.root, "des_tensor.pt")

        if not os.path.exists(path):
            description = np.load(os.path.join(self.root, 'description.npy'), allow_pickle=True)
            print('Loading RoBERTa')

            # Initialize the RoBERTa feature extraction pipeline
            feature_extraction = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base")
            des_vec = []

            # Embed each description using RoBERTa
            for each in tqdm(description):
                feature = torch.Tensor(feature_extraction(each))

                # Averaging the feature vectors
                feature_tensor = sum(feature[0]) / feature.shape[1]
                des_vec.append(feature_tensor)

            # Stack all feature tensors
            des_tensor = torch.stack(des_vec, 0).to(self.device)

            # Save the tensor if the save flag is set
            if self.save:
                torch.save(des_tensor, path)
        else:
            # Load the tensor if it already exists
            des_tensor = torch.load(path).to(self.device)

        print('Finished')
        return des_tensor
    
    def get_mentioned_mapping(self) -> list[list[int]]:
        """
        Creates a mapping of tweet IDs to mentioned user IDs in the dataset.

        Returns:
        List[List[int]]: A list of lists, where each inner list contains two integers 
                         representing a tweet ID and a mentioned user ID.
        """
        # Creating a mapping from usernames to user IDs
        username_mapping = self.df_data['profile'].apply(
            lambda x: {x['screen_name']: x['id']} if x is not None else {}
        ).to_dict()

        # List to store the mappings of tweet ID to mentioned user IDs
        tweet_mention_mapping = []

        # Iterate through each data entry
        for i in range(self.df_data.shape[0]):
            user_id = self.df_data.ID[i]  # ID of the user tweeting

            # Skip if there are no tweets
            if self.df_data['tweet'][i] is None:
                continue

            for tweet in self.df_data['tweet'][i]:
                # Extract all mentioned usernames in a tweet
                mentioned_usernames = extract_mentioned_usernames_in_tweet(tweet=tweet)

                # Update the list with mappings for each mentioned username
                if mentioned_usernames:
                    for mentioned_username in mentioned_usernames:
                        mentioned_id = username_mapping.get(mentioned_username)
                        if mentioned_id:
                            tweet_mention_mapping.append([user_id, mentioned_id])

        return tweet_mention_mapping

    def tweets_preprocess(self) -> np.ndarray:
        """
        Preprocesses the tweets feature of the dataset. It concatenates all tweets of a user into
        a single string. If the processed tweets are already saved, it loads them; otherwise, 
        it processes and saves them.

        Returns:
        np.ndarray: An array of concatenated tweet strings for each user.
        """
        print('Loading raw feature2...', end='   ')
        path = os.path.join(self.root, 'tweets.npy')

        if not os.path.exists(path):
            tweets = []

            # Iterate over each record in the dataset to process tweets
            for i in range(self.df_data.shape[0]):
                user_tweets = self.df_data['tweet'][i]

                # Concatenate all tweets of a user into a single string
                concatenated_tweets = '' if user_tweets is not None else None
                if concatenated_tweets is not None:
                    for tweet in user_tweets:
                        concatenated_tweets += str(tweet)

                tweets.append(concatenated_tweets)

            # Convert the list to a NumPy array
            tweets = np.array(tweets)

            # Save the processed tweets if the save flag is set
            if self.save:
                np.save(path, tweets)
        else:
            # Load the tweets if they have already been processed and saved
            tweets = np.load(path, allow_pickle=True)

        print('Finished')
        return tweets
    
    def tweets_embedding(self) -> torch.Tensor:
        """
        Processes and embeds the tweets feature of the dataset using the RoBERTa model.
        If the embeddings are already saved, it loads them; otherwise, it calculates and saves them.

        Returns:
        torch.Tensor: A tensor containing the embedded features of the tweets.
        """
        print('Running feature2 embedding')
        path = os.path.join(self.root, "tweets_tensor.pt")

        if not os.path.exists(path):
            tweets = np.load(os.path.join(self.root, 'tweets.npy'), allow_pickle=True)
            print('Loading RoBERTa')

            # Initialize the RoBERTa feature extraction pipeline
            feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', padding=True, truncation=True, max_length=500, add_special_tokens=True)
            tweets_list = []

            # Embed each tweet using RoBERTa
            for each_person_tweets in tqdm(tweets):
                total_each_person_tweets = torch.zeros(768).unsqueeze(0)  # Assuming RoBERTa-base output size
                ic
                valid_tweets_count = 0

                for each_tweet in each_person_tweets:
                    if each_tweet:  # Check if tweet is not empty
                        each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                        tweet_feature_avg = each_tweet_tensor.mean(dim=1)
                        total_each_person_tweets += tweet_feature_avg
                        valid_tweets_count += 1

                if valid_tweets_count > 0:
                    total_each_person_tweets /= valid_tweets_count

                tweets_list.append(total_each_person_tweets)

            tweet_tensor = torch.stack(tweets_list).to(self.device)

            # Save the tensor if the save flag is set
            if self.save:
                torch.save(tweet_tensor, path)
        else:
            # Load the tensor if it already exists
            tweet_tensor = torch.load(path).to(self.device)

        print('Finished')
        return tweet_tensor
    
    def num_prop_preprocess(self) -> torch.Tensor:
        """
        Processes numeric properties of user profiles in the dataset such as followers count,
        friends count, etc. It normalizes these features and combines them into a single tensor.
        If the processed data is already saved, it loads them; otherwise, it processes and saves them.

        Returns:
        torch.Tensor: A tensor containing normalized numeric properties of user profiles.
        """
        print('Processing feature3...', end='   ')
        num_properties_path = os.path.join(self.root, 'num_properties_tensor.pt')

        if not os.path.exists(num_properties_path):
            # Helper function to process and normalize each feature
            def process_and_normalize(feature):
                tensor = torch.tensor(np.array(feature, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(tensor, os.path.join(self.root, f'{feature}_count.pt'))
                return (tensor - tensor.mean()) / tensor.std()

            # Extracting and processing each numeric feature
            followers_count = [profile.get('followers_count', 0).strip() for profile in self.df_data['profile']]
            friends_count = [profile.get('friends_count', 0).strip() for profile in self.df_data['profile']]
            screen_name_length = [len(profile.get('screen_name', '')) for profile in self.df_data['profile']]
            favourites_count = [profile.get('favourites_count', 0).strip() for profile in self.df_data['profile']]
            statuses_count = [int(profile.get('statuses_count', 0)) for profile in self.df_data['profile']]

            # Special handling for active days
            reference_date = dt.strptime('Tue Sep 1 00:00:00 +0000 2020', '%a %b %d %X %z %Y')
            active_days = [(reference_date - dt.strptime(profile.get('created_at', reference_date.strftime('%a %b %d %X %z %Y')).strip(), '%a %b %d %X %z %Y')).days if profile else 0 for profile in self.df_data['profile']]

            # Normalize features
            followers_count = process_and_normalize(followers_count)
            friends_count = process_and_normalize(friends_count)
            screen_name_length = process_and_normalize(screen_name_length)
            favourites_count = process_and_normalize(favourites_count)
            active_days = process_and_normalize(active_days)
            statuses_count = process_and_normalize(statuses_count)

            # Concatenate all features into a single tensor
            num_prop = torch.cat([followers_count.unsqueeze(1), friends_count.unsqueeze(1), favourites_count.unsqueeze(1), statuses_count.unsqueeze(1), screen_name_length.unsqueeze(1), active_days.unsqueeze(1)], dim=1)

            if self.save:
                torch.save(num_prop, num_properties_path)
        else:
            num_prop = torch.load(num_properties_path).to(self.device)

        print('Finished')
        return num_prop
    
    def cat_prop_preprocess(self) -> torch.Tensor:
        """
        Processes categorical properties of user profiles in the dataset. Converts boolean 
        properties to numeric format (0 or 1). If the processed data is already saved, 
        it loads them; otherwise, it processes and saves them.

        Returns:
        torch.Tensor: A tensor containing the processed categorical properties.
        """
        print('Processing feature4...', end='   ')
        cat_properties_path = os.path.join(self.root, 'cat_properties_tensor.pt')

        if not os.path.exists(cat_properties_path):
            properties = ['protected', 'geo_enabled', 'verified', 'contributors_enabled', 
                          'is_translator', 'is_translation_enabled', 'profile_background_tile', 
                          'profile_use_background_image', 'has_extended_profile', 
                          'default_profile', 'default_profile_image']

            category_properties = []

            for profile in self.df_data['profile']:
                prop = [0] * len(properties) if profile is None else []
                for feature in properties:
                    # Convert boolean properties to 0 or 1
                    prop.append(1 if profile.get(feature) == "True" else 0)
                
                category_properties.append(np.array(prop, dtype=np.float32))

            # Convert the list to a tensor
            category_properties = torch.tensor(category_properties).to(self.device)

            if self.save:
                torch.save(category_properties, cat_properties_path)
        else:
            # Load the tensor if it already exists
            category_properties = torch.load(cat_properties_path).to(self.device)

        print('Finished')
        return category_properties
    
    def Build_Graph(self) -> (torch.Tensor, torch.Tensor):
        """
        Builds a graph structure from the Twibot20 dataset. It creates edges based on following,
        follower relationships, and tweet mentions.

        Returns:
        tuple(torch.Tensor, torch.Tensor): A tuple containing two tensors - edge_index and edge_type.
        - edge_index: Tensor of edge connections in the graph.
        - edge_type: Tensor indicating the type of each edge.
        """
        print('Building graph', end='   ')
        graph_path = os.path.join(self.root, 'edge_index.pt')
        edge_type_path = os.path.join(self.root, 'edge_type.pt')

        if not os.path.exists(graph_path):
            # Mapping from user ID to index in the dataset
            id2index_dict = {id: index for index, id in enumerate(self.df_data['ID'])}

            edge_index = []
            edge_type = []

            # Process following and follower relationships
            for i, relation in enumerate(self.df_data['neighbor']):
                if np.isnan(relation):
                    relation = None
                    
                if relation:
                    ic(relation)
                    # Following relationships
                    for each_id in relation['following']:
                        target_id = id2index_dict.get(int(each_id))
                        if target_id is not None:
                            edge_index.append([i, target_id])
                            edge_type.append(0)  # Type 0 for following

                    # Follower relationships
                    for each_id in relation['follower']:
                        target_id = id2index_dict.get(int(each_id))
                        if target_id is not None:
                            edge_index.append([i, target_id])
                            edge_type.append(1)  # Type 1 for follower

            # Add tweet mention relationships
            tweet_mention_mapping = self.get_mentioned_mapping()
            for pair in tweet_mention_mapping:
                edge_index.append(pair)
                edge_type.append(2)  # Type 2 for mentions

            # Convert to tensors and save
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)

            if self.save:
                torch.save(edge_index, graph_path)
                torch.save(edge_type, edge_type_path)
        else:
            # Load pre-existing graph data
            edge_index = torch.load(graph_path).to(self.device)
            edge_type = torch.load(edge_type_path).to(self.device)

        print('Finished')
        return edge_index, edge_type
    
    def train_val_test_mask(self) -> (range, range, range):
        """
        Generates index ranges for training, validation, and test datasets.

        Returns:
        tuple(range, range, range): A tuple containing three range objects for 
                                     the training, validation, and test datasets.
        """
        total = len(self.df_data)
        # Define the size of each dataset partition
        train_size = int(total*0.7)
        val_size = int(total*0.2)
        test_size = total - train_size - val_size

        # Create index ranges for each partition
        train_idx = range(train_size)
        val_idx = range(train_size, train_size + val_size)
        test_idx = range(train_size + val_size, train_size + val_size + test_size)

        return train_idx, val_idx, test_idx
    
    def dataloader(self):
            """
            Loads and processes various features of the Twibot20 dataset, preparing them for model training and evaluation.
            This includes embedding textual data, processing numeric and categorical properties, building a graph structure,
            and preparing index ranges for training, validation, and testing.

            Returns:
            tuple: A tuple containing tensors and index ranges for different features and dataset partitions.
                Includes tensors for description embeddings, tweet embeddings, numeric properties,
                categorical properties, graph edges, edge types, and labels, as well as index ranges
                for training, validation, and test datasets.
            """
            # Load labels
            labels = self.load_labels()

            # Preprocess description and tweets if necessary
            if self.process:
                self.Des_Preprocess()
                self.tweets_preprocess()

            # Embedding and processing various features
            des_tensor = self.Des_embbeding()
            tweets_tensor = self.tweets_embedding()
            num_prop = self.num_prop_preprocess()
            category_prop = self.cat_prop_preprocess()

            # Building the graph structure
            edge_index, edge_type = self.Build_Graph()

            # Generating index ranges for dataset partitions
            train_idx, val_idx, test_idx = self.train_val_test_mask()

            return des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx