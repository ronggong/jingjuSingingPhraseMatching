fs = 44100
# for the melodic similarity
framesize_melodicSimilarity = 2048
hopsize_melodicSimilarity = 1024

synthesizeLength = 5 # in second
sample_number_total = int(round(synthesizeLength * (fs / float(hopsize_melodicSimilarity))))

list_N_frames = [50,100,150,200,250,300,350,400,450,500,550,600]

# for the phonetic  similarity
framesize_phoneticSimilarity   = 1024
hopsize_phoneticSimilarity     = 512

framesize_t_phoneticSimiarity = framesize_phoneticSimilarity / float(fs)
hopsize_t_phoneticSimilarity = hopsize_phoneticSimilarity / float(fs)

# MFCC parameters
highFrequencyBound = fs/2 if fs/2<11000 else 11000
am = 'gmm'