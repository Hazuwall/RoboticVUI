""" ============================ Common ===================================="""
# Training
batch_size = 72
validation_size = 1020
stages = 2

# Raw WAV
framerate = 16000

# Spectrogram
spectrums_per_sec = 64
blur_size = 3

# Harmonics
min_fund_freq = 120
max_fund_freq = 300
harmonics_count = 32

# Acoustic Model
acoustic_model_name = "acoustic_model"
embedding_size = 80

# Classifier
classifier_name = "classifier"

# Datasets
cache_size = 2880
aug_rate = 0.3


""" ============================= Custom ====================================="""
