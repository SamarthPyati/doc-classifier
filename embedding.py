from langchain.evaluation.schema import PairwiseStringEvaluator
from sklearn.metrics.pairwise import cosine_similarity

from config import DEFAULT_RAG_CONFIG, RAGConfig, logger

class CosineEmbeddingEvaluator(PairwiseStringEvaluator):
    """ Custom evaluator for computing cosine similarity between embeddings """
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn

    def _evaluate_string_pairs(self, prediction, prediction_b, **kwargs):
        """ Evaluate similarity between two strings """
        try:
            vec_a = self.embedding_fn.embed_query(prediction)
            vec_b = self.embedding_fn.embed_query(prediction_b)
            sim = cosine_similarity([vec_a], [vec_b])[0][0]
            return {
                "score": sim,
                "explanation": f"Cosine similarity: {sim:.4f}"
            }
        except Exception as e:
            logger.error(f"Error evaluating string pairs: {e}")
            return {"score": 0.0, "explanation": f"Error: {e}"}
