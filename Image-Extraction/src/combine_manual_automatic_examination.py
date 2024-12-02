# manually inspected all lobbies of a streamer where first manual examination could not give results because later images were needed.
# Labeled according to following metric suggested by the images:
# - Delete -I: Even with later images role not detectable. Probably Lobby start is wrong timestamp for this streamer only (maybe streamer joined lobby late etc.) --> exclude this lobby for this streamer from analysis
# - Role name: for this lobby and this streamer the correct role was extracted and should now be assigned to him
import os
import pandas as pd
import pickle as pkl

second_manual_examination_results = pd.read_excel('ManualExamination_SecondStep.xlsx')

with open('identified_roles.pkl', 'rb') as file:
    identified_roles = pkl.load(file)

# reset role for all streamer lobbies where i have manual examination result
for sl in second_manual_examination_results['StreamerLobby']:
    print(sl)
    ses, l, streamer = sl.split(' - ')[0], sl.split(' - ')[1], sl.split(' - ')[2]
    if second_manual_examination_results[second_manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower() != 'delete -i':

        identified_roles[ses][l][streamer] = second_manual_examination_results[second_manual_examination_results['StreamerLobby'] == sl]['Role'].values[0].lower()
    # delete all other lobbies for streamer when role even with later images was not identifiable.
    else:
        if streamer in identified_roles[ses][l]:
            del (identified_roles[ses][l][streamer])

# last cleanup for all still existing delete - i roles
remove_entries = []
for ses in identified_roles.keys():
    for lob in identified_roles[ses].keys():
        for streamer in identified_roles[ses][lob].keys():
            if identified_roles[ses][lob][streamer] == 'delete - i':
                remove_entries.append([ses, lob, streamer])
for i in remove_entries:
    del(identified_roles[i[0]][i[1]][i[2]])


# Save the updated dictionary of identified roles
with open('identified_roles.pkl', 'wb') as f:
    pkl.dump(identified_roles, f)
