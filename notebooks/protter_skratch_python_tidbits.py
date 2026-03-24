# copying out circular data from within a notebook

out_dir = '/home/sosalab/local_repos/forks/phase-precession/data_for_testing_pycircstat_versions'
for day in multiDayData.keys():
    for ani in multiDayData[day].circ_licks.keys():
        for k, arr in multiDayData[day].circ_licks[ani].items():
            
            fname = f'day_{day}_{ani}_{k}_circ_licks'
            out_fpath = os.path.join(out_dir, fname)
            np.save(out_fpath, arr)

#  viewing some circular data
%matplotlib widget
fig, ax = plt.subplots()
for i, vals in enumerate(multiDayData[3].circ_licks['GCAMP2']['set 0']):
    ax.plot(vals)