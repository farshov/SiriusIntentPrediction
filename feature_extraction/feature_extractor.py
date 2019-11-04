from normalization import normalize


class FeatureExtractor:
    def corpus_extract_features(self, data, normalized_data=None):
        if not normalized_data:
            normalized_data = normalize(data)
        self.corpus_extract_features(data, normalized_data)

    def extract_features(self, data, normalized_data):
        pass

