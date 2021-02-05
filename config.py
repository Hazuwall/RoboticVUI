import dsp_utils
import models

""" ============================= Hot ======================================"""
# Training
hot_training_epochs = 1000
hot_training_batch_size = 72
hot_validation_size = 0
hot_test_size = 0
hot_display_step = 100
hot_dataset_index = 1

""" ============================ Manual ===================================="""
# Raw
framerate = 16000

# Spectrogram
step_count = 64
blur_size = 3

# Harmonics
min_fund_freq = 120
max_fund_freq = 300
harmonics_count = 32

# Datasets
dataset_index = hot_dataset_index
dataset_labels = ["u_ru_Shtooka", "s_en_SpeechCommands", "u_ru_CommonVoice", "t_ru_Shtooka",
                  "s_en_WikimediaCommons", "s_en_ShtookaEmmanuel", "s_en_ShtookaJudith", "s_en_ShtookaMary",
                  "s_en_ShtookaMaryNum", "s_ru_ShtookaGulnara", "s_ru_ShtookaSakhno", "s_ru_ShtookaNonFree",
                  "s_uk_ShtookaGalya", "s_uk_ShtookaSvitlana", "s_cz_ShtookaVeronika", "s_cz_ShtookaIvana",
                  "s_be_ShtookaIgor", "s_be_ShtookaDasha", "s_be_ShtookaJulia", "t_mx_Mix"]
dataset_label = dataset_labels[dataset_index]
datasets_root = "C:\\Datasets\\Audio\\"
dataset_cache = 2880
aug_rate = 0.3

# Training
training_epochs = hot_training_epochs
training_batch_size = hot_training_batch_size
validation_size = hot_validation_size
test_size = hot_test_size
display_step = hot_display_step
experiment_name = "lstm_encoder_60"
build = 1
models_root = "models\\"

# Encoder
encoder_step = None

# Clustering
embedding_overlap = 3
syllable_count = 3

# Recognition
words_dir = "words\\"


""" ============================= Auto ====================================="""
# Spectrogram
seg_length = dsp_utils.get_best_segment_length(framerate)
freq_count = dsp_utils.get_freq_count(seg_length)
freq_res = dsp_utils.get_freq_resolution(framerate, freq_count)

# Harmonics
preprocess_shape = (step_count, harmonics_count)


# Encoder
ACOUSTIC_MODEL_NAME = 'acoustic_model'
embedding_size = None


def set_embedding_size(size):
    global embedding_size
    embedding_size = size


set_embedding_size(60)
