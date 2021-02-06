import dsp_utils
import os

""" ============================ Common ===================================="""
# Experiment
experiment_name = "lstm_encoder_60"
build = 1
plugin_modules = ["config", "frontend", "models", "training"]

# Training
training_steps = 100
training_batch_size = 72
validation_size = 0
test_size = 0
display_interval = 100
checkpoint_interval = 100
verbose = False
models_root = "models\\"
if not verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Raw WAV
framerate = 16000

# Spectrogram
step_count = 64
blur_size = 3
seg_length = dsp_utils.get_best_segment_length(framerate)
freq_count = dsp_utils.get_freq_count(seg_length)
freq_res = dsp_utils.get_freq_resolution(framerate, freq_count)

# Harmonics
min_fund_freq = 120
max_fund_freq = 300
harmonics_count = 32
frontend_shape = (step_count, harmonics_count)

# Acoustic Model
acoustic_model_name = "acoustic_model"
embedding_size = None

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
datasets_root = "C:\\Datasets\\Audio\\"

# Recognition
min_word_weight = 0.75
silence_word = "_silence"
unknown_word = "_unknown"
ref_words_dir = "words\\"


""" ============================= Setters ====================================="""


def set_embedding_size(size: int):
    global embedding_size
    embedding_size = size


""" ============================= Custom ====================================="""
