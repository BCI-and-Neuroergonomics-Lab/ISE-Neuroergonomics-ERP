from tkinter.messagebox import askyesno
from tkinter import *
import matplotlib
import pandas
import mne
import os
mne.set_log_level(verbose="CRITICAL")  # hush MNE
Tk().withdraw()


def get_latency_amplitude(good_tmin, good_tmax, dat, ref=None, mode="abs"):
    #  NOTE: FOR UNKNOWN REASONS, ERP TIMESTAMPS NEED +1s ADDED ON (ex: 150ms -> 1150ms)
    good_tmax = good_tmax + 1.0
    good_tmin = good_tmin + 1.0

    erp = dat.copy()
    if ref:  # if a reference electrode is provided
        erp.pick([ref])  # focus on the one electrode
    stim = erp.average()

    try:
        _, lat = stim.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode=mode)  # gather peak latency
    except ValueError:
        return "NA", "NA"

    # Extract mean amplitude in µV over time
    stim.crop(tmin=good_tmin, tmax=good_tmax)
    mean_amp = stim.data.mean(axis=1)

    lat = int((lat - 1.0) * 1e3)  # convert latency to ms and remove the +1s
    amp = mean_amp[0] * 1e6  # grab our mean amplitude in µV
    return lat, amp


# load standard 10-20 for use later
montage = mne.channels.make_standard_montage('standard_1020')

# Format: Exp_Mediation_{}_vpxx.vhdr
format = dict(TeamA="Exp_Mediation_Paradigm1_Perception_vp",
              TeamB="Exp_Mediation_Paradigm4_Control_vp")
team = "TeamA"

P = {}
for i in range(1, 52):
    if i < 10:  # pad numbers <10 with one zero
        num = str(0) + str(i)
    else:
        num = str(i)
    fname = os.path.join("Data", format[team] + num + ".vhdr")
    try:
        P[num] = mne.io.read_raw_brainvision(fname)
    except FileNotFoundError:
        print("SUBJECT {} NOT FOUND".format(num))
        continue

    # 69 Channels total:
    # -65 scalp electrodes
    # -2 EOG electrodes (LE, RE)
    # -1 GSR electrode (GSR_MR_100_finger)
    # -1 ECG electrode (ECG)
    # Note: 3 unknown channels (Ne, Mat, Ext)
    # Referenced to FCz
    # Grounded at AFz
    # Sampled at 1000Hz
    # Filtered from 0.015Hz to 250Hz  *Note: MNE is reading these wrong, but not too important here

    new_types = []  # create a new channel types array
    for j in P[num].ch_names:
        if j == "LE" or j == "RE":  # mark left and right eye channels
            new_types.append("eog")
        elif "GSR" in j:  # mark GSR channel (won't be used)
            new_types.append("gsr")
        elif j == "ECG":  # mark our ECG channel
            new_types.append("ecg")
        elif j == "NE" or j == "Ma" or j == "Ext":  # mark misc channels, MA ?= mastoid, Ne ?= nasion, Ext ?= events
            new_types.append("misc")
        else:  # mark the rest as EEG channels from extended 10-20
            new_types.append("eeg")
    P[num].set_channel_types(dict(zip(P[num].ch_names, new_types)))  # apply new channel types to raw object
    P[num].set_montage(montage, on_missing="ignore")  # add standard 10-20 montage information for channel locations

# denote all events and their IDs
stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
mapping = {"Stimulus/S  5": 101, "Stimulus/S  6": 103, "Laser/L  1": 102, "New Segment/": 99999,
           "Stimulus/S  1": 1000, "Stimulus/S  2": 2000, "Stimulus/S  3": 3000}  # add the known stimulus labels
for i in range(0, 101):  # add the verbal pain ratings in a loop, as they can be anywhere from 0-100
    read = "Comment/" + str(i)
    write = i
    mapping[read] = write  # mimic the same format of {read this: change to this} for event annotations

# mark male vs female subjects
sex = dict(male=[2, 4, 5, 6, 9, 14, 15, 18, 19, 21, 22, 25, 27, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 48, 51],
           female=[1, 3, 7, 8, 10, 11, 12, 13, 16, 17, 20, 23, 24, 26, 28, 29, 30, 31, 32, 35, 37, 46, 47, 49, 50])

# important values for output file at the end
header = ["ID", "Sex", "Stimulus", "Component", "Value"]
fill = []

# begin processing the data!
for subject in P.keys():  # for each subject
    print("Subject: {}".format(subject))
    if int(subject) in sex["male"]:
        gender = "male"
    elif int(subject) in sex["female"]:
        gender = "female"
    data = P[subject]  # load the subject
    events, event_dict = mne.events_from_annotations(data, event_id=mapping)  # extract their events
    data.load_data()

    # Pre-Processing
    print("Filtering...")
    artifact_removal = data.copy()
    artifact_removal.filter(l_freq=1.0, h_freq=None, n_jobs=1)  # high-pass filter at 1Hz
    artifact_removal.notch_filter(50.0, n_jobs=1)  # notch filter at 50Hz

    # ICA artifact removal
    print("Fitting ICA...")
    ica = mne.preprocessing.ICA(n_components=10, random_state=42, max_iter="auto")
    ica.fit(artifact_removal)  # fit the ICA with EEG and EOG information

    # Visually inspect the data
    print("Visually inspecting components...")
    for i in range(ica.n_components_):  # look at each component
        ica.plot_properties(data, picks=[i], psd_args={"fmin": 1.0, "fmax": 60.0}, show=False)
        matplotlib.pyplot.show(block=False)
        if askyesno('ICA Component Analysis', 'Include Component?'):
            print("Component {} included".format(i))
        else:
            print("Component {} excluded".format(i))
            ica.exclude.append(i)
        matplotlib.pyplot.close()
    ica.apply(data)  # apply ICA to data, removing the artifacts

    # Epoch from -1500 to 3000ms. Should be 18 trials per stimulus intensity
    data.set_eeg_reference(ref_channels="average")
    # reject_criteria = dict(eeg=200e-6)  disabled rejection criteria to handle poor ICA extraction conditions
    all_epochs = mne.Epochs(data, events, event_id=event_dict, tmin=-1.5, tmax=3.0,
                            reject=None, preload=True, baseline=(-0.2, 0))

    for level in stims:  # for each stimulus level experienced
        ##################
        # ERP components #
        ##################
        print("Extracting ERPs for stimulus level {}".format(level[-1]))
        epochs = all_epochs.copy()[level]
        epochs.filter(l_freq=1.0, h_freq=30.0, n_jobs=1)  # 1-30Hz filter
        epochs.set_eeg_reference(ref_channels=["Fz"])  # re-reference to Fz

        # Get peak amplitude and latency of N1 at electrode C4
        latency, amplitude = get_latency_amplitude(0.150, 0.180, epochs, "C4", mode="neg")
        fill.append([subject, gender, level[-1], "N1_Lat", latency])
        fill.append([subject, gender, level[-1], "N1_Amp", amplitude])

        epochs.set_eeg_reference(ref_channels="average")  # re-reference to average

        # Get peak amplitude and latency of a baseline period
        _, amplitude = get_latency_amplitude(-1, 0, epochs)
        fill.append([subject, gender, level[-1], "Baseline_Amp", amplitude])

        # Get peak amplitude and latency of N2 at electrode CZ
        latency, amplitude = get_latency_amplitude(0.180, 0.210, epochs, "Cz", mode="neg")
        fill.append([subject, gender, level[-1], "N2_Lat", latency])
        fill.append([subject, gender, level[-1], "N2_Amp", amplitude])

        # Get peak amplitude and latency of P2 at electrode Cz
        latency, amplitude = get_latency_amplitude(0.290, 0.320, epochs, "Cz", mode="pos")
        fill.append([subject, gender, level[-1], "P2_Lat", latency])
        fill.append([subject, gender, level[-1], "P2_Amp", amplitude])

results = pandas.DataFrame(data=fill, columns=header)
results.to_csv("Output/TeamA_Results.csv")
