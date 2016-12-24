framesize = 2048
hopsize = 1024
fs = 44100
highFrequencyBound = fs/2 if fs/2<11000 else 11000
synthesizeLength = 5 # in second
sample_number_total = int(round(synthesizeLength*(fs/float(hopsize))))

list_N_frames = [50,100,150,200,250,300,350,400,450,500,550,600]