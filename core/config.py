""" ============================ Common ===================================="""
# Experiment
experiment_name = "lstm_encoder_60"
build = 1

# Training
training_batch_size = 72
validation_size = 0
test_size = 0
stages = 1

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
embedding_size = 60

# Classifier
classifier_name = "classifier"

# Datasets
dataset_index = 1
dataset_labels = ["u_ru_Shtooka", "s_en_SpeechCommands", "u_ru_CommonVoice", "t_ru_Shtooka",
                  "s_en_WikimediaCommons", "s_en_ShtookaEmmanuel", "s_en_ShtookaJudith", "s_en_ShtookaMary",
                  "s_en_ShtookaMaryNum", "s_ru_ShtookaGulnara", "s_ru_ShtookaSakhno", "s_ru_ShtookaNonFree",
                  "s_uk_ShtookaGalya", "s_uk_ShtookaSvitlana", "s_cz_ShtookaVeronika", "s_cz_ShtookaIvana",
                  "s_be_ShtookaIgor", "s_be_ShtookaDasha", "s_be_ShtookaJulia", "t_mx_Mix"]
dataset_label = dataset_labels[dataset_index]
cache_size = 2880
aug_rate = 0.3


""" ============================= Custom ====================================="""
