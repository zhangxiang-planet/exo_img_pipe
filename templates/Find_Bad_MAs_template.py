import os,glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
# from scipy.stats import norm, lognorm, gamma

def read_multiple_h5_files(base_dir):
    combined_amplitude_val = None
    combined_phase_val = None
    combined_freq = None
    ant = None
    pol = None

    msb = glob.glob(base_dir + '/MSB*.MS')
    msb.sort()

    for dir_name in msb:
        # dir_name = f"MSB{i:02d}.MS"
        file_path = os.path.join(dir_name, 'instrument_ddecal.h5')
        if os.path.isfile(file_path):
            with h5py.File(file_path, 'r') as h5_file:
                amplitude_group = h5_file['sol000/amplitude000']
                phase_group = h5_file['sol000/phase000']
                freq = amplitude_group['freq'][:]

                if combined_amplitude_val is None:
                    combined_amplitude_val = amplitude_group['val'][:]
                    combined_phase_val = phase_group['val'][:]
                    combined_freq = freq
                    ant = amplitude_group['ant'][:]
                    pol = amplitude_group['pol'][:]
                else:
                    combined_amplitude_val = np.concatenate((combined_amplitude_val, amplitude_group['val'][:]), axis=1)
                    combined_phase_val = np.concatenate((combined_phase_val, phase_group['val'][:]), axis=1)
                    combined_freq = np.concatenate((combined_freq, freq))

    return combined_amplitude_val, combined_phase_val, ant, combined_freq, pol

# replace read_multiple_h5_files with the following function if you want to read from a single file
def read_h5_file(base_dir):
    gsb = base_dir + '/GSB.MS'
    file_path = os.path.join(gsb, 'instrument_ddecal.h5')
    with h5py.File(file_path, 'r') as h5_file:
        amplitude_group = h5_file['sol000/amplitude000']
        phase_group = h5_file['sol000/phase000']
        freq = amplitude_group['freq'][:]

        amplitude_val = amplitude_group['val'][:]
        phase_val = phase_group['val'][:]
        ant = amplitude_group['ant'][:]
        pol = amplitude_group['pol'][:]

    return amplitude_val, phase_val, ant, freq, pol


def calculate_ratio(amplitude_val, pol):
    pol_labels = [p.decode() for p in pol]
    xx_index = pol_labels.index('XX')
    yy_index = pol_labels.index('YY')

    # Calculate the ratio of amplitudes of XX and YY
    ratio_val = amplitude_val[..., xx_index] / amplitude_val[..., yy_index]
    
    return ratio_val

def calculate_diff(phase_val, pol):
    pol_labels = [p.decode() for p in pol]
    xx_index = pol_labels.index('XX')
    yy_index = pol_labels.index('YY')

    # Calculate the ratio of amplitudes of XX and YY
    diff_val = phase_val[..., xx_index] - phase_val[..., yy_index]
    
    return diff_val

def plot_sol(val, ant, freq, pol, ylabel, output_filename, show_legend=True, highlight_antennas=None, log_y=False):
    n_ant = len(ant)
    # n_pol = len(pol)

    fig, axs = plt.subplots(int(np.ceil(n_ant/10)), 10, figsize=(20, 2 * int(np.ceil(n_ant/10))), constrained_layout=True, sharey=True)

    # Colors and markers for different polarizations
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    markers = ['o', 'x', '^', 'v']

    for i, ax in enumerate(axs.flat[:n_ant]):
        if np.isin(i, highlight_antennas):
            ax.set_facecolor((1, 0, 0, 0.1))

        for p, pol_value in enumerate(pol):
            if val.ndim == 5:
                data_to_plot = val[0, :, i, 0, p]
            elif val.ndim == 4:
                data_to_plot = val[0, :, i, p]

            alpha = 1

            ax.plot(freq, data_to_plot, color=colors[p], marker=markers[p % len(markers)], linestyle='', markersize=4, label=f'{pol_value.decode()}', alpha=alpha)
        ax.set_title(ant[i].decode())
        
        if log_y:
            ax.set_yscale('log')

        if i >= n_ant - 10:  # Only show x-axis ticks for the bottom row
            continue
        else:
            ax.set_xticks([])

    axs.flat[-1].legend()
    if not show_legend:
        axs.flat[-1].get_legend().remove()

    for ax in axs.flat[n_ant:]:
        ax.remove()

    fig.supxlabel('FREQUENCY')
    fig.supylabel(ylabel)
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

def find_bad_MAs(path_to_base_dir):
    # Replace 'path_to_base_dir' with the path to the base directory containing the SBxxx.MS folders
    # path_to_base_dir = './'
    # amplitude_val, phase_val, ant, freq, pol = read_multiple_h5_files(path_to_base_dir)
    amplitude_val, phase_val, ant, freq, pol = read_h5_file(path_to_base_dir)
    ratio_val = calculate_ratio(amplitude_val, pol)

    # take phase into consideration
    diff_val_phase = calculate_diff(phase_val, pol)

    # We need to put ratio_val on log scale, because 2 and 0.5 are equally bad
    ratio_val = np.log10(ratio_val)
    # ratio_val_phase = np.log10(ratio_val_phase)

    # Compute the standard deviation and mean along the frequency axis
    # medians = np.median(ratio_val, axis=0)
    # mads = np.median(np.abs(ratio_val - medians), axis=0)
    # modified_z_scores = np.where(mads != 0, 0.6745 * (ratio_val - medians) / mads, 0)
    # std_devs = np.std(ratio_val, axis=1)
    # means = np.mean(ratio_val, axis=1)

    # global_median = np.nanmedian(ratio_val[0,:,:,0])
    # global_mad = np.nanmedian(np.abs(ratio_val[0,:,:,0] - global_median))
    # z_scores = np.where(global_mad != 0, 0.6745 * (ratio_val[0,:,:,0] - global_median) / global_mad, 0)

    # counts = np.sum(np.abs(z_scores) > 3.5, axis=0)

    # Print the results
    # print("Antenna\tStandard Deviation\tMean Ratio")
    # print("Antenna\t\tMAD\tZ score")
    # print("---------------------------------------------")
    # for i, (std_dev, mean) in enumerate(zip(std_devs[0, :, 0], means[0, :, 0])):
    #     print(f"{ant[i].decode()}\t{std_dev:.6f}\t\t{mean:.6f}")

    # for i, (mad, modified_z_score) in enumerate(zip(mads[0, :, 0], modified_z_scores[0, :, 0])):
    #     print(f"{ant[i].decode()}\t{mad:.6f}\t\t{modified_z_score:.6f}")

    # Define thresholds
    # mean_threshold = 0.7  # Antennas with mean ratio outside (1 - mean_threshold, 1 + mean_threshold) will be considered bad
    # std_threshold = 0.07   # Antennas with a standard deviation larger than this value will be considered bad

    # Find bad antennas based on the mean and standard deviation thresholds
    # bad_antennas_mean = np.where((means[0, :, 0] < 1 / (1 + mean_threshold)) | (means[0, :, 0] > (1 + mean_threshold)))[0]
    # bad_antennas_std = np.where(std_devs[0, :, 0] > std_threshold)[0]

    # Combine the bad antennas found by both criteria
    # bad_antennas = np.unique(np.concatenate((bad_antennas_mean, bad_antennas_std)))
    # bad_antennas = np.where(counts > 0.1 * ratio_val[0,:,:,0].shape[0])

    # Change the method. A bad antenna has big deviation from a flat ratio_val spectrum
    variance = np.var(np.diff(ratio_val[0, :, :, 0], axis=0), axis=0)
    var_median = np.nanmedian(variance)
    var_mad = np.nanmedian(np.abs(variance - var_median))
    modified_z_scores = 0.6745 * (variance - var_median)/var_mad

    # find bad antennas with phase as well
    variance_phase = np.var(np.diff(diff_val_phase[0, :, :, 0], axis=0), axis=0)
    var_median_phase = np.nanmedian(variance_phase)
    var_mad_phase = np.nanmedian(np.abs(variance_phase - var_median_phase))
    modified_z_scores_phase = 0.6745 * (variance_phase - var_median_phase)/var_mad_phase
    modified_z_scores_phase_noremote = modified_z_scores_phase.copy()

    # however, we cannot use phase to judge if a remote antenna is bad. A remote antenna has ant_name matching 'MR1??NEN'
    # so we need to find the index of these antennas
    remote_antennas = np.where(np.char.find(ant, b'MR1') == 0)[0]
    modified_z_scores_phase_noremote[remote_antennas] = 0

    # bad antennas are those with either bad amplitude or phase
    # bad antennas have high modified_z_scores or modified_z_scores_phase_noremote

    # amp_threshold = np.nanmedian(np.abs(modified_z_scores)) * 10.3782   # equivalent to 7 sigma
    # phase_threshold = np.nanmedian(np.abs(modified_z_scores_phase_noremote)) * 10.3782   # equivalent to 7 sigma

    # replace the thresholds with 10 sigma
    amp_threshold = np.nanmedian(np.abs(modified_z_scores)) * 14.826   # equivalent to 10 sigma
    phase_threshold = np.nanmedian(np.abs(modified_z_scores_phase_noremote)) * 14.826   # equivalent to 10 sigma

    bad_antennas = np.where(np.logical_or(modified_z_scores > amp_threshold, modified_z_scores_phase_noremote > phase_threshold))[0]

    # Flag MR102NEN for quality reason, MR103NEN for UV distribution reasons

    # Try keeping MR102NEN and MR103NEN for now

    always_bad_antennas = [b'MR102NEN', b'MR103NEN']

    # Iterate over each always_bad_antenna
    for always_bad_antenna in always_bad_antennas:
        # Find the index of always_bad_antenna in the ant array
        index_always_bad = np.where(ant == always_bad_antenna)[0]

        # If always_bad_antenna is found in ant, add its index to bad_antennas
        if index_always_bad.size > 0:
            bad_antennas = np.unique(np.append(bad_antennas, index_always_bad))


    # print(bad_antennas)

    # Print the bad antennas
    # print("Bad Antennas:")
    # for i in bad_antennas:
    #     print(f"Antenna {ant[i].decode()}: Mean Ratio = {means[0, i, 0]:.6f}, Standard Deviation = {std_devs[0, i, 0]:.6f}")

    # print("Bad Antennas:")
    # for i in bad_antennas:
    #     print(f"Antenna {ant[i].decode()}: Median Ratio = {medians[0, i, 0]:.6f}, MAD = {mads[0, i, 0]:.6f}")

    with open(f'{path_to_base_dir}/bad_MA.txt', 'w') as file:
        print(','.join([ant_name.decode() for ant_name in ant[bad_antennas]]), file=file)


    # Plot the distribution of mean ratios
    # plt.figure(figsize=(8, 6))
    # plt.hist(means[0, :, 0], bins=np.logspace(np.log10(0.001), np.log10(10), 100), alpha=0.7)
    # plt.axvline(1 / (1 + mean_threshold), linestyle='--', color='r', label=f"Mean Threshold ({1 / (1 + mean_threshold):.2f}, {1 + mean_threshold:.2f})")
    # plt.axvline(1 + mean_threshold, linestyle='--', color='r')
    # plt.xscale('log')
    # plt.xlabel('Mean Ratio')
    # plt.ylabel('Number of Antennas')
    # plt.title('Distribution of Mean Ratios')
    # plt.legend()
    # plt.savefig('mean_distribution.png', dpi=300)

    # # Plot the distribution of standard deviations
    # plt.figure(figsize=(8, 6))
    # plt.hist(std_devs[0, :, 0], bins=100, alpha=0.7)
    # plt.axvline(std_threshold, linestyle='--', color='r', label=f"Std Threshold ({std_threshold:.2f})")
    # plt.xlabel('Standard Deviation')
    # plt.ylabel('Number of Antennas')
    # plt.title('Distribution of Standard Deviations')
    # plt.legend()
    # plt.savefig('std_distribution.png', dpi=300)

    # plot_sol(amplitude_val, ant, freq, pol, 'AMPLITUDE', 'amp_sol.png')
    # plot_sol(phase_val, ant, freq, pol, 'PHASE', 'phase_sol.png')
    # plot_sol(ratio_val, ant, freq, pol[:1], 'AMPLITUDE RATIO (XX/YY)', 'ratio_sol.png', show_legend=False, log_scale=True)

    plot_sol(amplitude_val, ant, freq, pol, 'AMPLITUDE', f'{path_to_base_dir}/amp_sol_highlighted.png', show_legend=True, highlight_antennas=bad_antennas)
    plot_sol(phase_val, ant, freq, pol, 'PHASE', f'{path_to_base_dir}/phase_sol_highlighted.png', show_legend=False, highlight_antennas=bad_antennas)
    plot_sol(ratio_val, ant, freq, pol[:1], 'AMPLITUDE RATIO (XX/YY)', f'{path_to_base_dir}/ratio_sol_highlighted.png', show_legend=False, highlight_antennas=bad_antennas)

    # add a plot for the phase ratio
    plot_sol(diff_val_phase, ant, freq, pol[:1], 'PHASE DIFF (XX-YY)', f'{path_to_base_dir}/diff_phase_sol_highlighted.png', show_legend=False, highlight_antennas=bad_antennas)

    bad_MA_names = ','.join([ant_name.decode() for ant_name in ant[bad_antennas]])

    return bad_MA_names
