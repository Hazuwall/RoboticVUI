build = 1

# Paths
experiments_dir = "D:\\Projects\\RoboticVUI\\vui\\vui\\experiments\\"
ref_words_dir = "D:\\Projects\\RoboticVUI\\vui\\vui\\resources\\ref_words\\"
test_words_dir = "D:\\Projects\\RoboticVUI\\vui\\vui\\resources\\test_words\\"
datasets_dir = "C:\\Datasets\\Audio\\"
checkpoint_prefix = "checkpoint-"

# Recognition
channels = 1
min_word_weight = 0.75
silence_word = "_silence"
unknown_word = "_unknown"

# Training
training_steps = 60000
display_interval = 100
checkpoint_interval = 1000
verbose = False
