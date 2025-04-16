# Author: Mainak Jas <mainakjas@gmail.com>

import numpy as np
import mne


def add_ch_loc(raws, info_helmet, ch_mapping):
    """Add channel locations.

    Parameters
    ----------
    raws : list
        List of raw objects.
    info_helmet : mne.Info
        Info object containing channel locations.
    ch_mapping : dict
        Dictionary containing channel mapping.
    """

    if not isinstance(raws, list):
        raws = [raws]

    n_runs = len(raws)

    def sanitize_helmet(ch_name):
        ch_label = int(ch_name.strip('A'))
        return f'MEG{ch_label:02d}'

    mne.channels.rename_channels(info_helmet, sanitize_helmet)

    for run in range(n_runs):

        raw = raws[run]

        # get inverse mapping
        inv_mapping = dict()
        for (k, v) in ch_mapping[f'run{run + 1}'].items():
            if v in raw.ch_names:

                if n_runs == 1:
                    inv_mapping[v] = f'MEG{k}'
                else:
                    inv_mapping[v] = f'MEG{k}_run{run + 1}'

        raw.rename_channels(inv_mapping)
        raw.reorder_channels(np.sort(raw.ch_names).tolist())

        for ch in raw.info['chs']:
            ch_name = None
            if ch['ch_name'] in info_helmet.ch_names:
                ch_name = ch['ch_name']
            elif ch['ch_name'].split('_')[0] in info_helmet.ch_names:
                ch_name = ch['ch_name'].split('_')[0]

            if ch_name is None:
                continue

            idx = info_helmet.ch_names.index(ch_name)
            ch['loc'] = info_helmet['chs'][idx]['loc']


def read_opm_info(fname):
    """Read info for OPM.

    Parameters
    ----------
    fname : str
        Path to the info file.
    """
    info = mne.io.read_info(fname)

    sel = mne.pick_channels_regexp(info['ch_names'], '(?!.*_index)')
    info = mne.pick_info(info, sel)

    for ch in info['chs']:
        if ch['coil_type'] == 3024:
            ch['coil_type'] = 8101
        ch['loc'][:3] /= 1000.  # convert to meters

    return info


def find_events(raw, **kwargs):
    """Find events in OPM-MEG data."""
    if not raw.preload:
        raise ValueError('raw data must be preloaded')

    if 'stim_channel' in kwargs:
        picks = mne.pick_channels(raw.ch_names, kwargs['stim_channel'])
    else:
        picks = mne.pick_types(raw.info, stim=True, meg=False, eeg=False)

    for pow, pick in enumerate(picks):
        stim_data = np.zeros_like(raw.times)
        st_data, times = raw[pick]
        st_data[st_data < 4] = 0.  # TTL pulses are 0-5 V
        st_data[st_data >=4] = 2 ** pow
        raw._data[pick] = st_data
    return mne.find_events(raw, **kwargs)


def get_good_chs(fnames):
    """Get good channels from txt files with residual fields.

    Each time on-sensor coils are used to zero the background field,
    the sensors which do not work may be different. This function will return
    the intersection of the good channels.

    Parameters
    ----------
    fnames : list
        List of file names.

    Returns
    -------
    good_chs : list
        The good channels.
    """
    good_ch_mask = None
    for fname in fnames:

        X = np.loadtxt(fname,
                       [('ch_name', 'U5'), ('Bx', '<f4'),
                        ('By', '<f4'), ('Bz', '<f4'), ('Cal', '<f4')])
        if good_ch_mask is None:
            good_ch_mask = ~np.isnan(X['Bx'])
        else:
            good_ch_mask = good_ch_mask & (~np.isnan(X['Bx']))
    return X['ch_name'][good_ch_mask]


def load_remnant_fields(fname, ch_names, bias=None):
    """Load remnant fields from txt file.

    Parameters
    ----------
    fname : str
        Path to the txt file.
    ch_names : list
        List of channel names.
    bias : dict
        Dictionary containing bias for each channel.
    """

    B = np.empty(len(ch_names,))

    X = np.loadtxt(fname,
                   [('ch_name', 'U5'), ('Bx', '<f4'),
                    ('By', '<f4'), ('Bz', '<f4'), ('Cal', '<f4')])
    for idx, ch_name in enumerate(ch_names):
        zdx = X['ch_name'].tolist().index(ch_name)
        B[idx] = X['Bz'][zdx] - bias[ch_name]

    return B
