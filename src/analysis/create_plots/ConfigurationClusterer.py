from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering


@dataclass
class CompactClusteringResults:
    """ייצוג קומפקטי של תוצאות הקלאסטרינג"""
    # שומר רק את המיפוי של config_id למספר קלאסטר
    assignments: Dict[str, int]
    # שומר רק את הפרמטרים החיוניים
    eps: float
    method: str

    @property
    def clusters(self) -> Dict[int, List[str]]:
        """מחזיר את הקלאסטרים המלאים רק כשצריך"""
        result = {}
        for config_id, cluster_id in self.assignments.items():
            if cluster_id not in result:
                result[cluster_id] = []
            result[cluster_id].append(config_id)
        return result


class ConfigurationClusterer:
    def _compute_hamming_distance_matrix(self, config_vectors: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """
        Computes pairwise Hamming distances among all configuration vectors.
        """
        config_ids = sorted(config_vectors.keys())
        vectors = [config_vectors[cid] for cid in config_ids]
        dist_array = pdist(vectors, metric='hamming')
        distance_matrix = squareform(dist_array)
        return config_ids, distance_matrix

    def cluster_configs(self,
                        config_vectors: Dict[str, np.ndarray],
                        method: str = 'dbscan',
                        eps: float = 0.3,
                        min_samples: int = 2) -> CompactClusteringResults:
        """
        מבצע קלאסטרינג ומחזיר תוצאות בפורמט קומפקטי
        """
        config_ids, distance_matrix = self._compute_hamming_distance_matrix(config_vectors)

        if method == 'dbscan':
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=eps,
                metric='precomputed',
                linkage='complete'
            )

        labels = clustering.fit_predict(distance_matrix)

        # שומר רק את המיפוי הבסיסי
        assignments = dict(zip(config_ids, labels))

        return CompactClusteringResults(
            assignments=assignments,
            eps=eps,
            method=method
        )

    def save_compact(self, results: CompactClusteringResults, filename: str):
        """שומר את התוצאות בפורמט קומפקטי"""
        np.savez_compressed(
            filename,
            assignments=np.array([(k, v) for k, v in results.assignments.items()]),
            eps=results.eps,
            method=results.method
        )

    def load_compact(self, filename: str) -> CompactClusteringResults:
        """טוען את התוצאות מהפורמט הקומפקטי"""
        with np.load(filename) as data:
            assignments = dict(data['assignments'])
            return CompactClusteringResults(
                assignments=assignments,
                eps=float(data['eps']),
                method=str(data['method'])
            )


if __name__ == "__main__":
    # דוגמה לשימוש
    config_vectors = {"config1": np.array([1, 1, 1, 0]), "config2": np.array([1, 0, 1, 0])}
    clusterer = ConfigurationClusterer()

    # ביצוע קלאסטרינג
    results = clusterer.cluster_configs(config_vectors)

    # שמירה בפורמט קומפקטי
    clusterer.save_compact(results, 'clusters_compact.npz')

    # טעינה חזרה
    loaded_results = clusterer.load_compact('clusters_compact.npz')

    # אם צריך לראות את הקלאסטרים המלאים
    clusters = loaded_results.clusters  # מחושב רק כשצריך

    # אם צריך לדעת לאיזה קלאסטר שייך config מסוים
    cluster_id = loaded_results.assignments['config1']
