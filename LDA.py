from time import time
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class LDA:
    def __init__(self, data=None):
        self.n_top_words = 5
        self.n_components = 7
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
        self.lda_model = LatentDirichletAllocation(self.n_components, max_iter=5, learning_method="online", random_state=42)
        self.model_fit(data) if data is not None else None

    def model_fit(self, data):
        print("Fitting LDA model...")
        start_time = time()
        X = self.vectorizer.fit_transform(data)
        self.lda_model.fit(X)
        end_time = time()
        print(f"LDA model fitted in {end_time - start_time:.2f} seconds")

    def get_topics(self):
        return self.lda_model.components_

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
    
    
    def plot_top_words(self, title):
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.get_topics()):
            top_features_ind = topic.argsort()[-self.n_top_words:]
            feature_names = self.get_feature_names()
            top_features = feature_names[top_features_ind]
            weights = topic[top_features_ind]

            print(f"Topic {topic_idx + 1}: {', '.join(top_features)}")
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 20})
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=30)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.5)
        plt.show()