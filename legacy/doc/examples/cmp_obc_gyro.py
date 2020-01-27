# Licensed under a 3-clause BSD style license - see LICENSE.rst
import Ska.engarchive.fetch as fetch
from Ska.Matplotlib import plot_cxctime
import Ska.Numpy

tstart = '2009:313:16:00:00'
tstop = '2009:313:17:00:00'

# Get OBC rates and gyro counts
obc = fetch.MSIDset(tstart, tstop, ['aorate1', 'aorate2', 'aorate3'], filter_bad=True)
gyr = fetch.MSIDset(tstart, tstop, ['aogyrct1', 'aogyrct2', 'aogyrct3', 'aogyrct4'], filter_bad=True)

# Transform delta gyro counts (4 channels) to a body rate (3 axes)
cts2rate = array([[-0.5       ,  0.5       ,  0.5       , -0.5       ],
                  [-0.25623091,  0.60975037, -0.25623091,  0.60975037],
                  [-0.55615682, -0.05620959, -0.55615682, -0.05620959]])

# Calculate raw spacecraft rate directly from gyro data
cts = np.array([gyr['aogyrct1'].vals,
                gyr['aogyrct2'].vals,
                gyr['aogyrct3'].vals,
                gyr['aogyrct4'].vals])
raw_times = (gyr['aogyrct1'].times[1:] + gyr['aogyrct1'].times[:-1]) / 2
delta_times = gyr['aogyrct1'].times[1:] - gyr['aogyrct1'].times[:-1]
delta_cts = cts[:, 1:] - cts[:, :-1]
raw_rates = np.dot(cts2rate, delta_cts) * 0.02 / delta_times

# Plot the OBC rates
figure(1, figsize=(8,6))
clf()
for frame, msid, label in ((1, 'aorate1', 'roll'),
                           (2, 'aorate2', 'pitch'),
                           (3, 'aorate3', 'yaw')):
    subplot(3, 1, frame)
    obc_rates = obc[msid].vals * 206254.
    plot_cxctime(obc[msid].times, obc_rates, '-')
    plot_cxctime(obc[msid].times, Ska.Numpy.smooth(obc_rates, window_len=20), '-r')
    ylim(average(obc_rates) + array([-1.5, 1.5]))
    title(label.capitalize() + ' rate (arcsec/sec)')

subplots_adjust(bottom=0.12, top=0.90)
# savefig('obc_rates_' + dur + '.png')

# Plot the S/C rates from raw gyro data
figure(2, figsize=(8,6))
clf()
for axis, label in ((0, 'roll'),
                    (1, 'pitch'),
                    (2, 'yaw')):
    subplot(3, 1, 1+axis)
    raw_rate = raw_rates[axis, :]
    plot_cxctime(raw_times, raw_rate, '-')
    plot_cxctime(raw_times, Ska.Numpy.smooth(raw_rate, window_len=20), '-r')
    ylim(np.mean(raw_rate) + np.array([-0.4, 0.4]))
    title(label.capitalize() + ' S/C rate (arcsec/sec)')

subplots_adjust(bottom=0.12, top=0.90)
# savefig('gyro_sc_rates_' + dur + '.png')

