import numpy as np
import tensorflow as tf
import datasets
import config as cfg
import models
from sklearn.cluster import MiniBatchKMeans

def main():
    words = ["zero", "one", "two", "three", "four"]
    avg(words)

def kmeans():
    provider = datasets.DatasetProvider(embeddings_return=True)
    kmeans = MiniBatchKMeans(n_clusters=cfg.syllable_count,random_state=0,
                             batch_size=cfg.training_batch_size)
    words = ["on", "stop", "tree"]
    print("Optimization Started!")
    for step in range(cfg.training_epochs):
        x, _ = provider.get_batch(cfg.training_batch_size,words)
        x = np.squeeze(x, axis=1)
        kmeans = kmeans.partial_fit(x)
    print("Optimization Finished!")
    centers = kmeans.cluster_centers_
    centers = np.expand_dims(centers.T, [0,1])
    model = models.Classifier(cfg.syllable_count)
    model.set_weights([centers, centers])
    model.save(cfg.training_epochs)

def avg(words):
    weights = np.zeros((1,cfg.embedding_features, len(words)))
    storage = datasets.HdfStorage(cfg.get_dataset_path('e'), "embeddings")
    i=0
    for word in words:
        codes = storage.fetch_subset(word,0,500, "RANDOM")
        codes = np.squeeze(codes,axis=1)
        weights[0,:,i] = np.mean(codes,axis=0)
        i+=1

    norm = np.linalg.norm(weights,axis=1,keepdims=True)
    weights /= norm
    model = models.Classifier(len(words))
    model.set_weights([weights])
    model.save(0)

if __name__ == "__main__":
    main() 