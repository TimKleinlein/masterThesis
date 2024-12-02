# Project Report
## Introduction
Social deduction games are an interesting field for answering research questions in the social sciences, as the players try to convince and deceive the other players through their communication and behavior in order to win the game. AmongUs is a well-known social deduction game that enjoys great popularity in the online gaming streaming community. With the goal of conducting social science analyses such as language analyses or predictions of player behavior, 363 VODs (video on demand streams) of streamers playing modifications of AmongUs with each other were downloaded from Twitch. The VODs come from 41 different sessions from games played in the period from 25.01.22 - 24.05.22. A [list of the specific modifications and versions of the game played](Lobby-Synchronization/data/streams_metadata/PlayedGames.xlsx) is provided in the Appendix. Several streamers play together in a session (there are an average of 8.9 VODs per session, although not all players have made their streams available for download, meaning that the actual number of players per session is probably slightly higher). A VOD is the recording of one session of one streamer. Several lobbies are played within a session. In order to use the available data for various analyses as previously outlined, it is essential to identify these individual lobbies and extract them from the VOD, as only these parts of the VOD show the behavior of the streamers in the game. It is also crucial to synchronize the lobbies of the different streamers participating in a session. Only if a temporal synchronization and therefore a certain simultaneity of the different streams is established, the behavior and statements of the streamers, which are largely influenced by the interaction with the other streamers, can be interpreted and analyzed meaningfully by adding the temporal context. These two tasks are carried out for the available data. The procedure is described in the following report.

## Data
Of the 394 VODs originally downloaded, only 363 VODs are used, as the session is only known for these. For each of these VODs there is an mkv file (video + audio) and an entry in a [VOD database](Lobby-Synchronization/data/streams_metadata/vods.db) containing metadata about the stream (published_at, session, title, chat, channel_id, resolution, duration, frame_count, memory_size, path, group, end_delta, start_delta).
In a prior step, an attempt was made to identify different lobbies for each of these VODs using image recognition: Individual frames were extracted for each VOD (at a rate of 3 frames per second), which were then converted to a 1280 x 720 pixel resolution. Since the lobby start and lobby end events in AmongUs each have a specific splash screen with a specific text, the extracted frames were examined for this pattern in order to identify the correct timestamp of these events. For this purpose, the frames were first prepared accordingly (identification of dark frames, zooming to the trigger locations, application of binary thresholding and morphological operations) before the text was analyzed with optical character recognition and, if applicable, the frame was assigned to one of the two events. The events identified in this way were saved for each VOD in a separate srt file ([examples of these srt files can be found here](Lobby-Synchronization/srt)). The srt file contains the start and end time for each identified event and thus a duration, as well as a text describing whether it is an identified lobby start or an identified lobby end.

## General procedure
The general procedure can be described as a three-part process. In a first step, the lobby start and end events identified by the image recognition (from the srt files) are combined with the existing metadata on the various VODs in order to identify lobbies for each session. For all streamers of the session, it is examined in which of these identified lobbies they participated. An attempt is then made to specify the corresponding timestamp in the streamer's VOD for these lobbies. Then this initial lobby assignment is further improved by applying various heuristics such that the assignment is already relatively successful. 
However, as it is not possible to make a definite correct assignment for individual streamers and lobbies in this way, a manual examination of the data is carried out in a second step. In this step, time stamps and assignments are provided for streamers and lobbies for which the purely programmatic assignment did not provide a sufficiently good result.
In a final step, the results from the programmatic lobby synchronization are supplemented by the results of the manual examination. In this way, all lobbies of each session are identified, as well as the corresponding timestamps of the streamers participating in those lobbies in their respective VODs.

## Programmatic lobby extraction and synchronization
In the following the steps taken for the [programmatic lobby extraction and synchronization](Lobby-Synchronization/src/initial_synchronization/synchronization.py) are outlined.
### Creation of lobby times
In a first step, arrays are formed for each streamer, which serve as an initial proposal for individual lobbies. These arrays are formed on the basis of the timestamps for lobby starts and lobby ends identified in the srt files and have the following form: [Possible lobby start, Possible lobby end]. This involves iterating through the chronologically ordered srt events of each streamer and filling the arrays one after the other. If a lobby start event follows another lobby start event or a lobby end event follows another lobby end event in the relevant srt file, the array is build with a None value. For example, two consecutive lobby start events followed by a lobby end event would create the following two arrays: [Lobby start, None], [Lobby start, Lobby end]. Srt events with a duration of less than 2 seconds or more than 30 seconds are skipped here, as it can be assumed that there has been an error in the image recognition. Arrays created that suggest a lobby duration of less than one minute are also removed, as such a short duration is very unrealistic and therefore indicates an error in the image recognition.  The arrays created for each streamer in the described way are called _lobby times_ in the following. Not all of the lobby times created will be assigned to a lobby later on. Since the comparison and matching of the lobby times of the individual streamers with the lobby times of the other streamers is carried out using UTC time, the srt times for potential lobby starts and ends are converted into UTC time. This works by adding the timestamps of the srt events to the start time taken from the metadata of the respective VODs (publish time plus start delta, which describes the time passed from the publish time to the start of the extracted part of the VOD).

### Creation of lobbies from lobby times
Based on the lobby times of the individual streamers participating in the session, initial proposals for lobbies are now being created. A _lobby_ is session-wide and the same for all streamers. The various streamers who have participated in the lobby have a lobby time for the lobby, which marks the timestamp of the respective lobby in their VOD. To create the lobbies for a session, in a first step all the lobby times of all the streamers in the session are sorted in ascending chronological order. Then, starting with the earliest lobby time, all lobby times are iterated over, and a decision is made for each lobby time as to whether it creates a new lobby or can be assigned to an existing lobby. The lobby time is always assigned to a lobby if the start time of the lobby time is a maximum of 2 minutes away from the start time of at least one of the lobby times already assigned to the lobby. If this is not the case, a new lobby is created. 
As only lobby times with a timestamp for the start and end time are used to create the lobbies, the lobby times with a None value are assigned in the next step. This involves calculating the median for the start and end times for all created lobbies based on all of the lobby times assigned to them. The lobby times with a None value are then assigned to a lobby whenever their existing time stamp is a maximum of 2 minutes away from the median start respectively median end of this lobby.
After the creation of the first lobby candidates, some streamers are excluded from further analyses, as lobby assignment is not possible for them. On the one hand, these are the streamers for which no srt file of the events from the image recognition is available (4 times the case in sessions used for the analysis, see Appendix). Secondly, streamers are removed if more than 50% of their lobby times are assigned to lobbies to which not a single other lobby time has been assigned.

### Adjustments to the created lobbies
After the creation of the first candidates for the lobbies, some adjustments are made to fix obvious errors such as unrealistically large numbers of created lobbies or unrealistic lobby durations. First of all, lobbies - which were created based on the start times of the lobby times - are now being merged based on their lobby end times. This means that all lobbies with a difference in the median end time of less than 45 seconds are merged with each other. 
Since the focus when creating the lobbies was on the start time of the lobby, there are some lobbies that are incorrectly ordered in temporal terms, as lobby n has a later end time than lobby n+1. This phenomenon can exist for two reasons: Firstly, a single lobby time in lobby n+1 has a too late lobby start with a correctly identified lobby end (ends slightly before the median end of lobby n) because the correct lobby start was not identified and has therefore created a new lobby n+1. In this case, the lobby time is assigned to lobby n and lobby n+1 is deleted. Secondly, a single lobby time after an incorrectly identified start has both an unidentified lobby end and a subsequent unidentified lobby start. This is problematic if the lobby time was the only lobby time that lobby n originally contained and therefore defines the median end alone. This is because all other lobby times added to lobby n have no identified start time and were therefore added solely on the basis of their end time, which means that all lobby times from n would actually have to be added to lobby n+1. In this case, the lobby time that caused the problem is split into two Lobby Times, one lobby time with the actual start time and an undefined end time, one lobby time with an undefined start time and the actual end time. The first lobby time is then assigned to lobby n, the second to lobby n+1. In addition, all lobby times previously assigned to lobby n are assigned to lobby n+1, and all lobby times previously assigned to lobby n+1 are assigned to lobby n.

### Improving the assignment of streamers to lobbies
In the assignment of the lobby times of the streamers to the lobbies, some heuristics are performed to further improve the assignment: For a streamer, all lobby times that have not been assigned to a lobby (this can only be the case for lobby times with a None value for start or end time) and that lie between two lobby times that have been assigned to two consecutive lobbies, are removed. In addition, if two lobby times have been assigned to the same lobby for a streamer, it is checked whether the first lobby time has no end time or the second lobby time has no start time. If this is the case, the two lobby times are merged by using the start time of the first lobby time and the end time of the second lobby time.

### Evaluation of the results and preparation for the manual examination
Following the programmatic lobby extraction and synchronization, meaningful results have already been achieved. However, some information is still missing for certain streamers and lobbies in order to make a final assignment. This missing information is collected through manual inspection of the video data. In order to minimize the effort of manual viewing as much as possible, a [log file](Lobby-Synchronization/data/initial_synchronization_output/Extraction) was created for each session, which summarizes the results of the programmatic procedure.
The file records for which streamers in the session the lobby assignment is non-consecutive, i.e. for which streamers the lobbies to which they were assigned do not correspond to a continuous sequence of numbers. This will then be used to investigate whether a lobby was really skipped by a streamer or was only skipped in the image recognition and programmatic evaluation. It should be noted at this point that the procedure described above registers a streamer joining the session later or leaving it earlier.
In addition, the _trustworthy lobby times_ are detected for the identified lobbies. A lobby time is considered trustworthy if at least one other lobby time assigned to the same lobby has a similar start and end time (maximum 10 seconds apart) and a similar duration (maximum 5 seconds difference). Here, only lobby times with a start and end time stamp are considered as candidates and as relevant comparison times.
Based on the trustworthy lobby times of the lobbies, it can also be concluded for which streamers trustworthy lobby times exist. All lobbies and streamers without a trustworthy lobby time are recorded in the log file.
Summarizing, in the final log file of each session for manual extraction, all streamers without an srt file, all streamers with non-consecutive lobbies, all lobbies without trustworthy lobby times, all streamers without trustworthy lobby times, as well as the start times of the individual lobbies for all streamers who participated in all lobbies (this serves for more effective manual processing) are recorded. Such an examplary log file for session 2022-01-27_S1 can be found [here](Lobby-Synchronization/data/initial_synchronization_output/Extraction/2022-01-27_S1.txt).

## Manual examination
In the manual examination, for each session the video data of all streamers for which no trustworthy lobby times exist are viewed. The video of the "main streamer", which is defined as the streamer who has participated in all lobbies of the session, is also viewed. For streamers without a trustworthy lobby time, a trustworthy lobby time is recorded for an arbitrary lobby. Based on the video of the main streamer, a trustworthy lobby time is recorded for each lobby without a trustworthy lobby time. In addition, this video is used to investigate whether the streamers with non-consecutive lobby assignments have actually skipped lobbies. To do this, In-game names are assigned to the various streamers to the best of my knowledge. A non-exhaustive list of name assignments is attached in the Appendix. In addition to the investigation of skipped lobbies, it is also investigated whether for streamers whose first or last lobby time was not assigned to a lobby, the None assignment can be replaced by existing lobbies in this case.
The results of the manual extraction are recorded in a [table](Lobby-Synchronization/data/manual_lobby_extraction/Results.xlsx) so that they can be merged with the results from the programmatic process ([streamers](Lobby-Synchronization/data/initial_synchronization_output/streamer_dictionaries), [lobbies](Lobby-Synchronization/data/initial_synchronization_output/lobbies_dictionaries), [participated lobbies by streamer](Lobby-Synchronization/data/initial_synchronization_output/assignedLobbiesDfs)) in a later step. The log files of 11 sessions suggest that these sessions should be excluded from the analysis, as the programmatic extraction could hardly produce any usable results. The affected sessions are listed in the following table, mostly the reason of the failure of the programmatic extraction was that for a majority of the streamers the srt file was missing and thus no trustworthy lobbies could be build.
 Excluded Sessions     |
 ----------- |
 2022-01-19_S1, 2022-01-20_S1, 2022-01-23_S1, 2022-01-23_S2, 2022-01-24_S1, 2022-02-03_S1, 2022-03-08_S1, 2022-01-30_S1, 2022-02-11_S1, 2022-03-06_S1, 2022-05-22_S1          |


## Combining the results from manual examination with the results from programmatic assigment
When merging the [manual results](Lobby-Synchronization/data/manual_lobby_extraction/Results.xlsx) with the [programmatic results](Lobby-Synchronization/data/initial_synchronization_output), all [streamers](Lobby-Synchronization/src/final_synchronization/streamer_dictionaries.py) and [lobbies](Lobby-Synchronization/src/final_synchronization/lobbies_dictionaries.py) that have not yet had a trustworthy lobby time are assigned one. 
Furthermore, the [lobby assignments](Lobby-Synchronization/src/final_synchronization/participated_lobbies.py), which define for each streamer which lobbies he has participated in, are finalized. First, for all streamers whose first or last lobby times were not assigned to a lobby, this lobby time is supplemented by the correct lobby assignment. Once the assignment of these special None values has been completed, all other None values are removed. In the next step, cases in which a streamer has been assigned to the same lobby several times are corrected by assigning the streamer to the lobby only once. Starting from this status, the earliest and latest lobby in which the streamer took part is identified for each streamer. It is then assumed for each streamer that they participated in all lobbies between these extreme values. This of course excludes the identified cases in which streamers have skipped individual lobbies, which is considered in the final lobby assignments for each streamer.

## Creating a streamer distance dictionary
At this point, you know which lobby each streamer has participated in. In addition, at least one trustworthy lobby time is known for each streamer and for each lobby. If the time offset between the start and end times of the different streamers is known, a start and end time can be calculated for each streamer for all lobbies in which they have participated. This is done by taking the known trustworthy lobby time for the lobby in question and calculating the start and end time, taking into account the time offset between the streamer and the streamer belonging to the trustworthy lobby time. How such a [dictionary, which represents the time differences between the streamers](Lobby-Synchronization/src/final_synchronization/streamer_dictionaries.py), is calculated, is explained in the following.
In a first step, all known temporal relationships between two streamers of a session are stored in a dictionary. A known temporal relationship is always given if the two streamers have a trustworthy lobby time for the same lobby, which does not differ in duration by more than 5 seconds. Since a known temporal relationship is recorded for all lobbies in which this is the case, it is possible that there are several known temporal relationships between a pair of streamers.
In the next step, based on this dictionary, in all sessions a network of temporal relationships could be created in which a temporal relationship can be derived between all streamer pairs of the session: either directly based on a known temporal relationship to each other or indirectly via a chain of known temporal relationships to intermediate streamers that connect the two streamers with each other. 
However, this type of network can only be created if the known temporal relationships are completely consistent. Due to possible inconsistencies in the image recognition (certain lobby events were not recognized in the first but in a later image, due to the extraction rate of 3 fps there are small temporal inconsistencies even when the first image is recognized, ...) or in the timestamp of the stream (small broadcasting problems may lead to streams lagging and thus differences in the temporal differences between streamers), this is not given in the present case. An attempt must thus be made to create a network of temporal relationships that is the most likely possible, taking into account the known temporal relationships. In order to achieve this, the problem at hand is treated as a linear system of equations of the form <p align="center">
  <img src="https://latex.codecogs.com/svg.image?&space;Ax=b" alt="Your Equation">
</p>

The matrix _A e R\_(nxs)_, with _n_ being the number of known temporal relations, _s_ being the number of streamers in the session, consists exclusively of the values {1, -1, 0} and acts as a selection matrix: it represents a known temporal relation in each of its rows by selecting the two affected streamers by the values {-1, 1}, while the remaining streamers are ignored by {0}. The vector _x e R\_(s)_ contains all streamers, with the streamers involved in the known temporal relationship being selected by multiplication with the selection matrix. The vector _b e R\_(n)_ represents the temporal differences between the two selected streamers of the known temporal relationship. It should be noted that if two streamers have several known temporal relationships (because they both have a trustworthy lobby time in the same lobby multiple times), the average value of the temporal difference is taken. In this way, all known temporal relationships are displayed. In the next step, the most likely temporal relationships between the streamers are calculated on this basis. The least square solution is calculated to solve the system of equations _Ax=b_, which calculates a vector _x_ that minimizes the 2-norm |b - Ax|. By subtracting the timestamps obtained for the streamers by the optimization from the vector _x_, the most likely temporal relationship to each other can now be calculated for each pair of streamers.

## Merging the results obtained to create the final lobby assignments
After creating the streamer distance dictionary, the [final lobby times can now be calculated](Lobby-Synchronization/src/final_synchronization/combine_results.py) for each streamer as mentioned before. In addition to the [streamer distance dictionary](Lobby-Synchronization/data/final_synchronization_output/streamer_offsets.pkl), the information about [which lobbies the streamers have participated in](Lobby-Synchronization/data/final_synchronization_output/final_lobby_assignments.pkl) and the [trustworthy times of the individual lobbies](Lobby-Synchronization/data/final_synchronization_output/trustworthy_lobbies.pkl) are used as mentioned: One iterates through all lobbies of the session and calculates the lobby time for each streamer who has participated in the lobby by adding the time difference between the streamer and the streamer from whom the trustworthy lobby time originates to the trustworthy lobby time of the lobby. If a lobby has several trustworthy lobby times from several streamers, the lobby time of the streamer with the most trustworthy lobby times in the other lobbies of the session is used for the calculation. In a final step their respective [start time](Lobby-Synchronization/data/final_synchronization_output/streamer_start_times.pkl) is subtracted from the streamers lobby times to get the timestamps in srt format, before the [dictionary with the final lobbies](Lobby-Synchronization/data/final_synchronization_output/final_lobby_times.pkl) is saved.

## Evaluation of lobby extraction and synchronization
The final lobby times assigned to the streamers are [evaluated](Lobby-Synchronization/src/evaluation/evaluation.py) in a final step. This involves randomly selecting 10 lobbies from all sessions and streamers and manually checking their correctness. The [results](Lobby-Synchronization/data/evaluation/Evaluation.xlsx), which are shown below, indicate a successful extraction, as nine of the ten selected lobby times actually belong to the correct lobby number. In one case, a lobby was not recognized in the extraction process, which is why the times of lobby 15 actually correspond to lobby 16. When evaluating the nine successfully assigned lobby times, the identified start point of the lobby is on average 1.5 seconds away from the actual start point, while the identified end is on average even only 0.1 seconds away.

| Streamer   | Session       | Lobby | Start    | End      | Correct Start | Correct End |
| ---------- | ------------- | ----- | -------- | -------- | ------------- | ----------- |
| jvckk      | 2022-03-01_S1 | 12    | 01:57:08 | 02:01:45 | 01:57:07      | 02:01:45    |
| aribunnie  | 2022-02-24_S1 | 6     | 00:50:21 | 00:54:40 | 00:50:23      | 00:54:40    |
| karacorvus | 2022-01-25_S1 | 15    | 02:42:02 | 02:47:13 | 02:33:19      | 02:41:20    |
| kaywordley | 2022-02-22_S1 | 11    | 01:31:55 | 01:39:41 | 01:31:55      | 01:39:41    |
| reenyy     | 2022-03-10_S1 | 10    | 01:42:33 | 01:52:15 | 01:42:34      | 01:52:15    |
| vikramafc  | 2022-01-27_S1 | 13    | 01:47:05 | 02:02:03 | 01:47:02      | 02:02:04    |
| skadj      | 2022-03-03_S1 | 8     | 01:07:28 | 01:14:07 | 01:07:30      | 01:14:07    |
| irepptar   | 2022-01-26_S1 | 6     | 00:55:52 | 01:02:45 | 00:55:50      | 01:02:45    |
| vikramafc  | 2022-01-27_S1 | 4     | 00:26:10 | 00:33:50 | 00:26:13      | 00:33:50    |
| vikramafc  | 2022-03-09_S1 | 3     | 00:23:50 | 00:29:14 | 00:23:50      | 00:29:14    |

## Future work
All in all, a well-functioning lobby extraction and synchronization was achieved with the described procedure. In a future project, an attempt will be made to further improve the lobby synchronization. One idea is to use the AmongUs feature that a characteristic sound is played at lobby start and lobby end events in addition to the splash screen described above. This sound can possibly be used to achieve an even more precise synchronization based on the already extracted and synchronized lobbies of the various streamers.

## Appendix
### Played modifications and versions of AmongUs
| Game                   | Count |
| ---------------------- | ----- |
| TheOtherRoles v3.3.3   | 2     |
| TheOtherRoles v3.4.0   | 2     |
| TownOfUs v2.5.0 dev6   | 2     |
| TheOtherRoles v3.4.2   | 3     |
| TownOfUs v2.5.0 dev12  | 2     |
| TownOfUs  v2.5.0 dev14 | 4     |
| TownOfUs  v2.5.1s      | 4     |
| TheOtherRoles v3.4.3   | 1     |
| TownOfUs  v2.6.0s      | 4     |
| TownOfUs  v2.6.1s      | 5     |
| TheOtherRoles v3.4.4   | 1     |
| TownOfUs  v2.6.4s      | 5     |
| TheOtherRoles v4.1.3   | 1     |
| TownOfUs  v3.0.1s      | 2     |
| TownOfUs  v3.1.0s      | 2     |

### Streamers without srt file in sessions used in the analysis
| Session       | Streamer    |
| ------------- | ----------- |
| 2022-01-28_S1 | vikramafc   |
| 2022-02-01_S1 | thecasperuk |
| 2022-03-01_S1 | elainaexe   |
| 2022-05-24_S1 | aribunnie   |

Note that for multiple streamers of sessions which were not included in the analysis the srt file was also missing. This is also for many of them the reason why the programmatic extraction did not work.

### Assigned In-Game Names for streamer
| Streamer          | In-Game Name                |
| ----------------- | --------------------------- |
| uneasypeasy       | Uneasy, Unevsy              |
| irepptar          | Rep                         |
| br00d             | br00d, Ovalbrood, Cvalbrood |
| aribunnie         | AriBunnie                   |
| skadj             | Skadj, SKvdj                |
| vikramafc         | VikramAFC                   |
|                   | PJONK                       |
| karacorvus        | Kara, Kvrv                  |
| x33n              | X33N                        |
| ozzaworld         | Ozza, Ozzv                  |
| cheesybluenips    | Cheesy                      |
| jvckk             | jvckk                       |
| zeroyalviking     | Ze                          |
| willyutv          | Will                        |
| junkyard          | Junk                        |
| pwuppygf          | pwuppygf                    |
| paulleewhirl      |                             |
|                   | Casper                      |
| reenyy            | Reenyy                      |
|                   | SquatchyLS                  |
|                   | Dunrunnin                   |
|                   | Atlas                       |
|                   | NerdOut                     |
|                   | Chilled                     |
| kaywordley        | Kay                         |
| tenmamaemi        | TenmaMaemi                  |
| dooleynotedgaming | Jeremy                      |
| pastaroniravioli  | Pasta, Pastaroni            |
| falcone           | Falcone                     |
|                   | Sparkly                     |
| therealshab       | Shab                        |
|                   | Karasper                    |
| atla5_w1nz        | ATLA5                       |
| ayanehylo         | Hylo                        |
|                   | Dun                         |
| brizzynewindsong  | Brizzyne                    |
|                   | sophietx                    |
| ressnie           | Ressnie, Rvssniv            |
|                   | Elswick                     |
|                   | WOLFY                       |
|                   | Rawbuhbuh                   |
|                   | Nortski                     |
|                   | ZERO                        |
| jojosolos         | jvjvsvlvs                   |
|                   | sophimane                   |
| chey              | Chey                        |
| taydertot         | TayderTot                   |
| aplatypuss        | APIatypus                   |
| kruzadar          | Kruzadar                    |
| dumbdog           | DumbDog                     |
| kyr_sp33dy        | Speedy                      |
| sidearms4reason   | SideArms                    |
| itsdanpizza       | Dan Pizza                   |
| heckmuffins       | HeckMufins                  |
| jayfletcher88     | Jay                         |
|                   | Noor                        |
|                   | crunchy                     |
| hcjustin          | HCJustin                    |
| paulleewhirl      | Paullee, PaulLee            |
| courtilly         | Covrtilly                   |
|                   | Juggernaut                  |
