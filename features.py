# Author        : Simon Bross
# Date          : January 20, 2024
# Python version: 3.6.13


import numpy as np
from tqdm import tqdm
from textblob import TextBlob
from collections.abc import Iterable
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
import pattern.text.en as pattern
from sentence_transformers import SentenceTransformer
# nltk.download('averaged_perceptron_tagger')


class FeatureExtractor:
    """
    A class for extracting various linguistic features from textual data.
    The features comprise a binary question feature, POS tags, mood, modality,
    sentiment, subjectivity/objectivity, and sentence embeddings.
    """
    def __init__(self, train_data, test_data):
        """
        Initializes the FeatureExtractor with training and test data.
        @param train_data: Training dataset as a list of strings (sentences).
        @param test_data: Test dataset as a list of strings (sentences).
        """
        # ensure correct data constitution
        assert isinstance(train_data, Iterable)
        for sent in train_data:
            assert isinstance(sent, str)
        assert isinstance(test_data, Iterable)
        for sent in test_data:
            assert isinstance(sent, str)
        self.__train_data = train_data
        self.__test_data = test_data

    @property
    def train_data(self):
        """
        Getter to access the training data.
        @return: Training data as a list of strings (sentences).
        """
        return self.__train_data

    @property
    def test_data(self):
        """
        Getter to access the test data.
        @return: Test data as a list of strings (sentences).
        """
        return self.__test_data

    def get_features(self, select="all"):
        """
        Extracts various features from the data for the training and test set.
        This method allows the extraction of different features based on
        the specified parameters. Features include:
            - 'question': Binary feature matrix for question sentences.
            - 'pos': Count vectorized matrix for part-of-speech tags.
            - 'mood': Matrix representing the mood of sentences.
            - 'modality': Modality matrix representing the modality of
               sentences.
            - 'subjectivity': Subjectivity matrix representing the
               subjectivity/objectivity of sentences.
            - 'sentiment': Sentiment matrix representing the
               sentiment scores of sentences.
            - 'embedding': Sentence embeddings matrix.
        @param select: Defaults to "all", thus extracting every available
               feature. Alternatively, a list of strings can be provided
               to select a specific subset of the available features.
        @return: A tuple containing two NumPy arrays:
            - feature_matrix_train (numpy.ndarray): Feature matrix for the
              training data, where each row corresponds to a sentence,
              and each column to a feature.
            - feature_matrix_test (numpy.ndarray): Feature matrix for the test
              data, where each row corresponds to a sentence, and each column
              to a feature.
        """
        features = [
            "question", "pos", "mood", "modality", "subjectivity",
            "sentiment", "embedding"
        ]
        # collect feature arrays for train and test data
        feature_arrays_train = []
        feature_arrays_test = []
        if select != "all":
            assert isinstance(select, list), \
                "'Select' parameter must be set to 'all' or provide a list" \
                " of strings corresponding to the desired features "
            # check if features in 'select' are valid
            assert all(
                [True if feature in features else False for feature in select]
            ), "Invalid feature(s) found in 'select'"
            # sort out features that do not need to be computed
            features = [feature for feature in select]
        for feature in tqdm(
                features,
                desc=f"Extracting {len(features)} feature(s) from the data"
        ):
            feature_train, feature_test = getattr(self, '_' + feature)()
            feature_arrays_train.append(feature_train)
            feature_arrays_test.append(feature_test)
        # concatenate all features to a single feature matrix
        feature_matrix_train = np.concatenate(feature_arrays_train, axis=1)
        feature_matrix_test = np.concatenate(feature_arrays_test, axis=1)
        return feature_matrix_train, feature_matrix_test

    def _question(self):
        """
        Extracts a binary feature for whether a sentence is a question or not.
        The binary feature is set to 1 if a sentence contains a question mark,
        and 0 otherwise.
        @return: A tuple containing two two-dimensional Numpy arrays:
           - question_feature_train: Binary feature matrix for question
             sentences in the training data, where each row corresponds
             to a sentence.
           - question_feature_test: Binary feature matrix for question
             sentences in the test data, where each row corresponds to
             a sentence.
        """
        question_feature_train = np.array(
            [[1] if "?" in sent else [0] for sent in self.train_data]
        )
        question_feature_test = np.array(
            [[1] if "?" in sent else [0] for sent in self.test_data]
        )
        return question_feature_train, question_feature_test

    @staticmethod
    def __tokenize_and_pos_tag(sent):
        """
        Helper function that tokenizes and performs part-of-speech tagging
        (Penn Treebank tagset) on a given sentence.
        @param sent: Input sentence (as str) to be tokenized and POS tagged.
        @return: A list of tuples where every element contains a token along
        with its respective POS tag.
        """
        tokenized_sent = word_tokenize(sent)
        tagged_sent = pos_tag(tokenized_sent)
        return tagged_sent

    def _pos(self):
        """
        Applies POS-tagging on the training and test data. The frequency
        of every POS tag in a sentence is counted using count-vectorization.
        @return: Count-vectorized POS matrix for the training and test data,
        as np.ndarray.
        """
        train_processed = map(self.__tokenize_and_pos_tag, self.train_data)
        test_processed = map(self.__tokenize_and_pos_tag, self.test_data)
        # extract POS tags from structure [(string, tag) , ... ]
        only_tags_train = [
            [tag for word, tag in sent] for sent in train_processed
        ]
        only_tags_test = [
            [tag for word, tag in sent] for sent in test_processed
        ]
        # vectorize the POS tags (only_tags) using CountVectorizer
        # as only_tags is already tokenized, CountVectorizer does not need to
        # perform tokenization
        cv = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        matrix_train = cv.fit_transform(only_tags_train).toarray()
        matrix_test = cv.transform(only_tags_test).toarray()
        return matrix_train, matrix_test

    @staticmethod
    def __mood_to_val(mood_list):
        """
        Convert a list of mood labels into corresponding numerical values based
        on their relevance for the discrimination between CFS (Check-worthy
        Factual Sentence), UFS (Unimportant Factual Sentence), and NFS
        (Non-Factual Sentence). The numerical values are determined by the
        following mapping:
            - 'indicative': 2 (Likely to contain factual claims)
            - 'imperative': 1 (Less likely to express a factual claim)
            - 'conditional': 0 (The proposition made via conditional mood is
                               dependent on some condition, thus unfactual
            - 'subjunctive': 0 (Likely to rather express opinions, feelings,
                                or beliefs)
        @param mood_list: A list of mood labels for sentences.
        @return: A list where each inner list contains a numerical value
        corresponding to the mood of a sentence.
        """
        mapping = {
            'indicative': 2,
            'imperative': 1,
            'conditional': 0,
            'subjunctive': 0
        }
        converted = [
            [mapping[mood_str]] for mood_str in mood_list
        ]
        return converted

    def _mood(self):
        """
        Analyzes the grammatical mood of sentences using the Pattern library.
        Converts the mood labels (cf. __mood_to_val method) into corresponding
        numerical values for each sentence in the training and test data.
        @return: A tuple containing two NumPy arrays:
            - mood_train (numpy.ndarray): Mood matrix for the training data,
              one row per sentence.
            - mood_test (numpy.ndarray): Mood matrix for the test data,
              one row per sentence.
        """
        # sentences need to be parsed for further processing
        parsed_train = [
            pattern.parse(sent, lemmata=True) for sent in self.train_data
        ]
        parsed_test = [
            pattern.parse(sent, lemmata=True) for sent in self.test_data
        ]
        # convert parsed sentences into Sentence objects and get their mood
        mood_train = list(
            map(pattern.mood, map(pattern.Sentence, parsed_train))
        )
        mood_test = list(
            map(pattern.mood, map(pattern.Sentence, parsed_test))
        )
        # convert mood (e.g. 'indicative') into its respective numerical value
        # by means of a helper function
        mood_train = np.array(self.__mood_to_val(mood_train))
        mood_test = np.array(self.__mood_to_val(mood_test))
        return mood_train, mood_test

    def _modality(self):
        """
        Analyzes modality (expression of possibility and necessity) using the
        Pattern library. Computes the degree of certainty as values between
        [-1, 1] for the training and test data, where values greater than
        0.5 are expected to represent facts (cf. Pattern library).
        @return: A tuple containing two NumPy arrays:
            - modality_train (numpy.ndarray): Modality matrix for the training
              data, one row per sentence.
            - modality_test (numpy.ndarray): Modality matrix for the test
              data, one row per sentence.
        """
        modality_train = map(pattern.modality, self.train_data)
        modality_test = map(pattern.modality, self.test_data)
        # nest list so that every item corresponds to one sentence
        modality_train = np.array([[mod] for mod in modality_train])
        modality_test = np.array([[mod] for mod in modality_test])
        return modality_train, modality_test

    def _subjectivity(self):
        """
        Analyzes subjectivity using the TextBlob library. Computes the
        subjectivity score within the range [0,1] for each sentence in
        the training and test data, where 0.0 is very objective and 1.0
        very subjective.
        @return: A tuple containing two NumPy arrays:
            - subj_train (numpy.ndarray): Subjectivity matrix for the training
              data, one row per sentence.
            - subj_test (numpy.ndarray): Subjectivity matrix for the test
              data, one row per sentence.
        """
        subj_train = [
            [blob.sentiment_assessments.subjectivity] for blob in
            [TextBlob(text) for text in self.train_data]
        ]
        subj_test = [
            [blob.sentiment_assessments.subjectivity] for blob in
            [TextBlob(text) for text in self.test_data]
        ]
        return np.array(subj_train), np.array(subj_test)

    def _sentiment(self):
        """
        Analyzes sentiment using the TextBlob library. Computes the sentiment
        score ranging from [-1, 1] for each sentence in the training and
        test data.
        @return: A tuple containing two elements:
           - sentiment_train (numpy.ndarray): Sentiment matrix for the training
             data, one row per sentence.
           - sentiment_test (numpy.ndarray): Sentiment matrix for the test
             data, one row per sentence.
        """
        sentiment_train = [
            [blob.sentiment_assessments.polarity] for blob in
            [TextBlob(sent) for sent in self.train_data]
        ]
        sentiment_test = [
            [blob.sentiment_assessments.polarity] for blob in
            [TextBlob(sent) for sent in self.test_data]
        ]
        sentiment_train = np.array(sentiment_train)
        sentiment_test = np.array(sentiment_test)
        return sentiment_train, sentiment_test

    def _embedding(self):
        """
        Generate sentence embeddings using the 'all-MiniLM-L12-v2' model
        from the sentence transformers library to encode the training
        and test data into dense (384 dimensions) vector representations.
        @return: A tuple containing two elements:
                - train_embedds (numpy.ndarray): Embedding matrix for the
                  training data, one row per sentence.
                - test_embedds (numpy.ndarray): Embedding matrix for the
                  test data, one row per sentence.
        """
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        train_embedds = model.encode(self.train_data, convert_to_numpy=True)
        test_embedds = model.encode(self.test_data, convert_to_numpy=True)
        return train_embedds, test_embedds
