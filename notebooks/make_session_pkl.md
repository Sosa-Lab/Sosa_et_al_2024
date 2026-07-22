---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
  kernelspec:
    display_name: dope2p
    language: python
    name: python3
---

# Make sess class from raw data for each session and save as pickle

Run after suite2p curation but before anything else.

Currently uses info from sessions_dict.py to loop through sessions and create the sess class, \
synchronizing neural data with behavioral data.

sess pickle files will be named `<scene>_<session>_<scan>.pickle`  \
and saved in `path_dict['preprocessed_root']/sess/<animal>/<date>`.

Set `overwrite` to `True` if you want to overwrite existing .pickle files. Otherwise, you will get an error that the file already exists.

```python
overwrite = False
```

```python
import os
import numpy as np

from reward_relative import preprocessing as pp
from reward_relative import utilities as ut

import TwoPUtils


%load_ext autoreload
%autoreload 2
```

### Specify your path dictionary here.

Copy and rename `path_dict.py` to a new file and edit it with the paths on your system.

```python
from reward_relative.path_dict_msosa_mac import path_dictionary as path_dict
path_dict
```

## Scroll or click to the desired section for:

[Behavior only](#Behavior-only)



Within each section, define animal and iterate through sessions.


While running the below cells, if you get an error that says `DatabaseError: Execution failed on sql 'SELECT * FROM data': no such table: data`,
check that all of your .sqlite files are named properly and have data in them (i.e. `Scene_1.sqlite` instead of `'Scene_1(1).sqlite'`



# Behavior only

```python
from reward_relative.sessions_dict_behavior_only import sosalab as metadata
```

```python
metadata
```

```python
## Define animal
animal = 'pp01'
days = np.arange(0, len(metadata[animal])) # range of days
# days =days[1:3] # optional select subset of days
days
```

### Main cell to create sess

```python
basedir = os.path.join(path_dict['preprocessed_root'], animal)
sbxdir = os.path.join(path_dict['sbx_root'], animal)
vrdir = path_dict['VR_Data']

binary_from_sbxdir = False # only relevant for downsampling
calcium_exists = False

load_suite2p = False
load_scaninfo = False
VR_only = True

trial_matrix_kwargs = []

for i, day in enumerate(days):

    if type(metadata[animal][day]) is not tuple:
        date = metadata[animal][day]['date']
        scene = metadata[animal][day]['scene']
        rig = metadata[animal][day]['rig']
        session = metadata[animal][day]['session']
        scan_number = metadata[animal][day]['scan']

        sess = pp.create_sess(basedir, sbxdir, vrdir, animal, date, rig, scene, session, scan_number,
                              load_scaninfo=load_scaninfo,
                              load_VR=True,
                              load_suite2p=load_suite2p,
                              load_behavior=True,
                              VR_only=VR_only,                              
                              )

        sess_dir = os.path.join(
            path_dict['preprocessed_root'], 'sess', animal, date)
        os.makedirs(sess_dir, exist_ok=True)
        print(sess_dir)

        if np.isnan(scan_number):
            scan_number=0
            
        sess_name = '%s_%03d_%03d.pickle' % (scene,
                                             session,
                                             scan_number
                                             )
        # Write sess to pickle file
        ut.write_sess_pickle(sess, sess_dir, sess_name, overwrite=overwrite)

    else:
        print("Iterating through multiple sessions")
        for i in range(len(metadata[animal][day])):
            date = metadata[animal][day][i]['date']
            scene = metadata[animal][day][i]['scene']
            session = metadata[animal][day][i]['session']
            scan_number = metadata[animal][day][i]['scan']

            sess = pp.create_sess(basedir, sbxdir, vrdir, animal, date, scene, session, scan_number,
                                  load_scaninfo=True,
                                  load_VR=True,
                                  load_suite2p=True,
                                  load_behavior=True)

            sess_dir = os.path.join(
                path_dict['preprocessed_root'], 'sess', animal, date)
            os.makedirs(sess_dir, exist_ok=True)
            print(sess_dir)

            sess_name = '%s_%03d_%03d.pickle' % (scene,
                                                 session,
                                                 scan_number,
                                                 )
            # Write sess to pickle file
            ut.write_sess_pickle(
                sess, sess_dir, sess_name, overwrite=overwrite)
```

```python
sess = ut.load_sess_pickle(path_dict['preprocessed_root'], 'pp01', exp_day=1)
```

```python

```
