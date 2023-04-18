import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic
from keybert import KeyBERT

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import json
#----------------------------------------------------------------------
def topic_labeling(doc_df, num_of_topics):
	docs_of_topic_temp = [doc_df[doc_df.topic_id==topic_idx].doc.tolist() for topic_idx in range(num_of_topics)]
	docs_of_topic_list = ["\n".join(_) for _ in docs_of_topic_temp] #joined docs of each topic.

	kw_model = KeyBERT() #initialize keybert model.
	keywords = kw_model.extract_keywords(docs_of_topic_list, keyphrase_ngram_range=(1, 3))

	topic_label_options_list = [{ii:a for ii, (a, b) in enumerate(keywords[topic_idx])} for topic_idx in range(num_of_topics)] #list of dict of label options.
	topic_label_list = [topic_label_options[0] for topic_label_options in topic_label_options_list] #list of labels.

	return topic_label_list, topic_label_options_list
#----------------------------------------------------------------------
class TopicModel:
	#----------------------------------------------------------------------
	def __init__(self):
		self.module_name = "TopicModel"
	#----------------------------------------------------------------------
	def via_bertopic(self, doc_list, generate_topic_labels=False):
	
		#Embedding
		sentence_model = SentenceTransformer("all-MiniLM-L6-v2") #!!! choose which embedding to do.
		embeddings = sentence_model.encode(doc_list, show_progress_bar=False)

		# Dimensionality reduction:
		umap_random_state = 238 #!!!needs to not be put in manually!!!
		umap_model = UMAP(random_state=umap_random_state, verbose=False)

		#Topic Model
		topic_model = BERTopic(calculate_probabilities=True,
								diversity=0.2, #MMR
								n_gram_range=(1,3), min_topic_size=20,
								umap_model=umap_model,
								embedding_model=sentence_model)

		topics, probabilities = topic_model.fit_transform(doc_list)


		topic_df = topic_model.get_topic_info() #This is being used to find out number of topics. There must be a better way of finding number of topics from the topic_model object itself. !!!
		num_of_topics = topic_df["Topic"].values.max() + 1

		doc_list_array = np.array(doc_list)
		
		topic_df = pd.DataFrame(
			{
				"topic_id" : list(range(num_of_topics)),
				"topic_size": topic_df[topic_df.Topic>=0].Count.values,
			}
		)

		doc_df = pd.DataFrame(
			{
				"doc": doc_list,
				"topic_id": topic_model.topics_,
				# "topic_id_score": [max(prob) for prob in topic_model.probabilities_]
			}
		)

		#Topic labeling:
		if generate_topic_labels==True:
			topic_label_list, topic_label_options_list = topic_labeling(doc_df=doc_df, num_of_topics=num_of_topics)
			topic_df["topic_name"] = topic_label_list
			topic_df["topic_name_options"] = topic_label_options_list

		return topic_df, doc_df
	#----------------------------------------------------------------------
	def via_LDA(self, doc_list, generate_topic_labels=False, num_of_topics=5, random_state=42):
		"""
		Parameters
		----------
		doc_list: list
			List of documents
		num_of_topics: int, default=5
			Number of topics
		generate_topic_labels: bool, default=False
			If True, will generate topic labels
		random_state: int, default=42
			Int or reproducible results across multiple function calls
		"""
		tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english", ngram_range=(1,2))
		tf = tf_vectorizer.fit_transform(doc_list)
		
		lda = LatentDirichletAllocation(n_components=num_of_topics, random_state=random_state).fit(tf)
		
		score = lda.transform(tf)
		topic_id_list = np.argmax(score, axis=1)

		doc_df = pd.DataFrame(
				{
					"doc": doc_list,
					"topic_id": topic_id_list
					# "topic_id_score": np.max(score, axis=1)
				}
			)
		
		topic_df = pd.DataFrame(
				{
					"topic_id" : list(range(num_of_topics)),
					"topic_size": [doc_df[doc_df.topic_id==topic_idx].shape[0] for topic_idx in range(num_of_topics)]
				}
			)
		
		#Topic labeling:
		if generate_topic_labels==True:
			topic_label_list, topic_label_options_list = topic_labeling(doc_df=doc_df, num_of_topics=num_of_topics)
			topic_df["topic_name"] = topic_label_list
			topic_df["topic_name_options"] = topic_label_options_list
		
		return topic_df, doc_df
#----------------------------------------------------------------------