import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
#----------------------------------------------------------------------
class SentimentAnalysis:
	#----------------------------------------------------------------------
	def __init__(self):
		self.module_name = "SentimentAnalysis"
		
		task = 'sentiment'
		self.model_path = f'cardiffnlp/twitter-roberta-base-{task}'
		self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
		self.model.save_pretrained(self.model_path)
		self.tokenizer.save_pretrained(self.model_path)
	#----------------------------------------------------------------------
	def short_input(self, doc_list):
		pipe = pipeline(task='sentiment-analysis', model=self.model, tokenizer=self.tokenizer)
		senti_dict_list = pipe(doc_list)

		label_list = [senti_dict["label"] for senti_dict in senti_dict_list]
		score_list = [senti_dict["score"] for senti_dict in senti_dict_list]

		sentiment_df = pd.DataFrame(
			{
				"doc" : doc_list,
				"sentiment": label_list,
				"sentiment_score": score_list
			}
		)

		sentiment_label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
		sentiment_df['sentiment'] = sentiment_df['sentiment'].map(sentiment_label_map)
		
		return sentiment_df
	#----------------------------------------------------------------------

