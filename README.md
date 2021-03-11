# Music-Coding-for Music Generation

# Dataset
We will use very tiny dataset made with only one MIDI file (data/original_metheny.mid). Instead of using musical notes directly, we will use high-level representations that encode musical features of jazz music (chord progression, beats, and so on) in the input as a dictionary. As a result, 60 integer sequences made of length of 30 are generated from the data MIDI. Each integer indicates the index of encoding dictionary. The number of dictionary is 78 (constants.N_DICT).
