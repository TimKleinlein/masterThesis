import pickle as pkl

with open('../Image-Extraction/identified_roles.pkl', 'rb') as file:
    roles = pkl.load(file)

with open('../Speech-Extraction/PVAD_Application/data/final_transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)


remove_dr = []
for dr in transcriptions.keys():
    role_found = False
    session = dr[:13]
    lob = dr.split('_')[2][1:]
    streamer = dr.split('_')[4]
    # disc_round = dr.split('_')[3]

    if lob in roles[session].keys():
        for s in roles[session][lob].keys():
            if streamer == s.split('_')[2]:
                transcriptions[dr]['role'] = roles[session][lob][s]
                role_found = True

    if not role_found:
        remove_dr.append(dr)

for dr in remove_dr:
    del (transcriptions[dr])

# save role transcription dictionary in data
with open('data/transcriptions.pkl', 'wb') as f:
    pkl.dump(transcriptions, f)
