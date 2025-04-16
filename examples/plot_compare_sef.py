"""
=============================
06. Analyze SEF Data with OPM
=============================

This example demonstrates how to analyze somatosensory evoked field (SEF) data
collected with optically pumped magnetometers (OPM) and compare it with
superconducting quantum interference device (SQUID) data. The OPM data was
obtained with the background field nulling coils we developed.
"""

# Author: Mainak Jas <mjas@mgh.harvard.edu>

import json
import matplotlib.pyplot as plt

import mne
from opmcoils.analysis import find_events, add_ch_loc, read_opm_info

from pathlib import Path
import pooch

# sphinx_gallery_thumbnail_number = 3

# %%
# First, use pooch to download the data

url = "https://osf.io/aqh27/download"

target_dir = Path.cwd() / 'data'
project_dir = target_dir / 'sef_data'

path = pooch.retrieve(
    url=url,
    known_hash=None,  # We don't know the hash
    path=target_dir,
    fname='sub-01.zip',  # Specify the filename
    processor=pooch.Unzip(extract_dir=project_dir),  # Extract to a folder named 'sub-01'
    progressbar=True
)

# %%
# Then, we define the file paths

subject = 'sub-01'
date = '20240722'
cond = 'median'

subjects_dir = project_dir / subject
raw_fname = subjects_dir / 'opm_meg' / date / f'{date}_{subject}_paneltesting_{cond}_raw.fif'
raw_fname_squid = subjects_dir / 'squid_meg' / date / 'RT_median_raw.fif'
helmet_info_fname = target_dir / 'helmet_99channel_size-60.fif'
ch_mapping_fname = subjects_dir / 'opm_meg' / date / 'mapping.json'

# %%
# Then, we start by processing OPM data
bads = ['01:15', '00:14', '00:15', '00:16']

raw_opm = mne.io.read_raw_fif(raw_fname, preload=True)
raw_opm.rename_channels(lambda x: x.strip('-BZ_CL'))
raw_opm.info['bads'] = bads

raw_opm.set_channel_types({'Input-1': 'stim'})

# %%
# Electrical median nerve stimulation creates artifacts. We
# will annotate the artifacts to avoid filtering the sharp transients.
events = find_events(raw_opm, min_duration=1. / raw_opm.info['sfreq'])
annot = mne.annotations_from_events(events, raw_opm.info['sfreq'],
                                    event_desc={1: 'BAD_STIM'})
# XXX: annotations_from_events should have duration option
annot.onset -= 0.002
annot.duration += 0.003

raw_opm.notch_filter(60.)

raw_opm.set_annotations(annot)
raw_opm.filter(4., None, skip_by_annotation=())
raw_opm.filter(None, 150, skip_by_annotation=('BAD_STIM'))

# %%
# Then, we create evoked response
reject = None # dict(mag=11e-12)
epochs_opm = mne.Epochs(raw_opm, events, tmax=0.3,  # ISI = 300 ms
                        reject_by_annotation=False,
                        reject=reject, reject_tmin=0.03,
                        preload=True)
evoked_opm = epochs_opm.average()

# %%
# We add channel locations for the OPM sensors
# and plot the evoked response with spatial colors.
info_helmet = read_opm_info(helmet_info_fname)

with open(ch_mapping_fname, 'r') as fd:
    ch_mapping = json.load(fd)

add_ch_loc(evoked_opm, info_helmet, ch_mapping)

evoked_opm.plot()

# %%
# We use HFC to remove background artifacts.
projs_hfc = mne.preprocessing.compute_proj_hfc(evoked_opm.info, order=1)
evoked_opm.add_proj(projs_hfc)
evoked_opm.apply_proj()

# Alternatively, one could use SSP:
# projs_ssp = mne.compute_proj_epochs(epochs_opm.copy().crop(tmax=0), n_mag=2)
# evoked_opm.add_proj(projs_ssp)
# evoked_opm.apply_proj()

# %%
# Let us now process the SQUID data.
raw_squid = mne.io.read_raw_fif(raw_fname_squid, raw_fname_squid, preload=True)
raw_squid.pick_types(meg='mag', stim=True)
events = mne.find_events(raw_squid, min_duration=1. / raw_opm.info['sfreq'])

annot = mne.annotations_from_events(events, raw_squid.info['sfreq'],
                                    event_desc={1: 'BAD_STIM'})
# XXX: annotations_from_events should have duration option
annot.onset -= 0.002
annot.duration += 0.003
raw_squid.set_annotations(annot)

raw_squid.filter(4., None)
raw_squid.notch_filter(60.)
raw_squid.filter(None, 150, skip_by_annotation=('BAD_STIM'))
epochs = mne.Epochs(raw_squid, events, tmax=0.3,
                    reject_by_annotation=False)
evoked_squid = epochs.average()
evoked_squid.plot(ylim=dict(mag=(-300, 300)))

# %%
# "Butterfly plots" are not appropriate to compare the
# evoked responses between OPMs and SQUID because the sensor
# locations are not comparable and hide individual sensor time series.
#
# That is why, we pick sensors in similar locations on the
# OPM-MEG and SQUID-MEG and compare the evoked responses.
squid_chs = ['MEG0431', 'MEG0311']
opm_chs = ['MEG30', 'MEG09']
scale = 1e15
fig, axes = plt.subplots(2, 1, sharex=True)

for i, (opm_ch, squid_ch) in enumerate(zip(opm_chs, squid_chs)):
    ax = axes[i].twinx()
    ln1 = axes[i].plot(evoked_opm.times * 1e3,
                       evoked_opm.copy().pick([opm_ch]).data[0] * scale,
                       '#e66101')
    ln2 = ax.plot(evoked_squid.times * 1e3,
                  evoked_squid.copy().pick([squid_ch]).data[0] * scale,
                  '#5e3c99')
    axes[i].set_title(f'Location #{i + 1}')
    axes[i].legend(ln1 + ln2, ['OPM', 'SQUID'], loc=1)
    ax.legend()

fig.text(0.97, 0.45, 'SQUID data (fT)', va='center',
         rotation='vertical', fontsize=12)
fig.text(0.02, 0.45, 'OPM data (fT)', va='center',
         rotation='vertical', fontsize=12)
axes[1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()

# Add inset
squid_ch, opm_ch = squid_chs[0], opm_chs[0]
inset_ax = axes[0].inset_axes([0.25, 0.5, 0.08, 0.45],
                              xlim=(10, 30), ylim=(-150, 200),
                              xticks=[], yticks=[])
inset_ax.plot(evoked_opm.times * 1e3,
              evoked_opm.copy().pick([opm_ch]).data[0] * scale,
              '#e66101')
inset_ax.plot(evoked_squid.times * 1e3,
              evoked_squid.copy().pick([squid_ch]).data[0] * scale,
              '#5e3c99')
for spine in inset_ax.spines.values():
    spine.set_linestyle('dotted')
axes[0].indicate_inset_zoom(inset_ax, edgecolor='black', linestyle='dotted')
