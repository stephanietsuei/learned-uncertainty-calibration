import numpy as np
from scipy.stats import chi2, rv_histogram
from scipy.signal import periodogram
import matplotlib.pyplot as plt


def chi2_overlay(nees, df, title, plt_chi2_pdf=True, chi2_percentile=0.999,
                 bin_width=0.5, plot_nbins=None, num_line_pts=2000):

  # Get the Chi-2 pdf
  x = np.linspace(0, chi2.ppf(chi2_percentile,df), num_line_pts)
  rv = chi2(df)
  chi2_pdf = rv.pdf(x)

  # Get the last pt to use in the histogram
  if plt_chi2_pdf:
    last_pt = np.ceil(x[-1])
  else:
    last_pt = np.ceil(np.max(nees))

  # Get bins for a pretty-looking histogram
  if plot_nbins == "auto":
    bins = "auto"
  elif plot_nbins is not None:
    bins = np.linspace(0, last_pt, plot_nbins)
  else:
    bins = np.arange(0, last_pt, bin_width)

  # Plot histogram against chi2 pdf
  plt.figure()
  plt.title(title)
  #plt.xlabel(r"$\rho_k$")
  #plt.ylabel(r"PDF")

  plt.hist(nees, density=True, bins=bins, alpha=0.5, edgecolor='black')

  if plt_chi2_pdf:
    plt.plot(x, chi2_pdf)
    plt.xlim(0, x[-1])


def interval_bounds_plot(nees_sum, df, alpha, title, nruns):

  rv = chi2(df*nruns)
  (lb,ub) = rv.interval(alpha)

  nees = nees_sum / nruns
  lb = lb / nruns
  ub = ub / nruns

  num_timesteps = nees.shape[0]

  plt.figure()
  plt.plot(lb*np.ones(num_timesteps,), 'r')
  plt.plot(ub*np.ones(num_timesteps,), 'r')
  plt.plot(nees, 'b')
  plt.title(title)
  plt.xlabel("Timesteps")
  plt.ylabel("Average NEES")


def sigma_hist(sigmas, title, plot_nbins=10):
  plt.figure()
  plt.title(title)
  plt.xlabel(r"# $\sigma$")
  plt.ylabel("Timesteps in Bin")
  plt.hist(sigmas, bins=plot_nbins, alpha=0.5, edgecolor='black')


def time_plot(time_axis, signal, title=None, ylabel=None, xlabel=None):
  plt.figure()
  _time_plot(time_axis, signal, title=title, ylabel=ylabel, xlabel=xlabel)


def _time_plot(time_axis, signal, title=None, ylabel=None, xlabel=None):
  plt.plot(time_axis, signal)
  if xlabel is not None:
    plt.xlabel(xlabel)
  if ylabel is not None:
    plt.ylabel(ylabel)
  if title is not None:
    plt.title(title)


def time_three_plots(time_axis, signals, suptitle, titles=None, xlabel=None):
  if titles is None:
    titles = [None, None, None]
  
  plt.figure()
  plt.suptitle(suptitle)
  plt.subplot(3,1,1)
  _time_plot(time_axis, signals[0,:], title=titles[0], ylabel="x-axis")
  plt.subplot(3,1,2)
  _time_plot(time_axis, signals[1,:], title=titles[1], ylabel="y-axis")
  plt.subplot(3,1,3)
  _time_plot(time_axis, signals[2,:], title=titles[2], ylabel="z-axis",
    xlabel=xlabel)


def error_three_plots(time_axis, error_signals, seq, error_type, error_unit):
  means = np.mean(error_signals, axis=1)
  var = np.var(error_signals, axis=1)
  print("{} {} error means: {} {}".format(seq, error_type, means, error_unit))
  print("{} {} error variance: {}".format(seq, error_type, var))

  plt.figure()
  plt.suptitle("{} {} error ({})".format(seq, error_type, error_unit))
  plt.subplot(3,1,1)
  _time_plot(time_axis, error_signals[0,:],
    "x-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[0], var[0]))
  plt.subplot(3,1,2)
  _time_plot(time_axis, error_signals[1,:],
    "y-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[1], var[1]))
  plt.subplot(3,1,3)
  _time_plot(time_axis, error_signals[2,:],
    "z-axis error mean/var: {0:10.3g}, {1:10.3g}".format(means[2], var[2]))


def plot_3D_error_cloud(error_signal, title):
  fig = plt.figure()
  plt.suptitle(title)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(error_signal[0,:], error_signal[1,:], error_signal[2,:],
    marker='.')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')


def psd_plot(inn_signal, sample_freq, suptitle):
  f_inn, psd_inn = periodogram(inn_signal, sample_freq, axis=1)
  psd_inn = np.sqrt(psd_inn)
  plt.figure()
  plt.suptitle(suptitle)
  plt.subplot(3,1,1)
  plt.semilogy(f_inn, psd_inn[0,:])
  plt.ylabel("x")
  plt.subplot(3,1,2)
  plt.semilogy(f_inn, psd_inn[1,:])
  plt.ylabel("y")
  plt.subplot(3,1,3)
  plt.semilogy(f_inn, psd_inn[2,:])
  plt.ylabel("z")
  plt.xlabel("Frequency (Hz)")


def plot_3d_trajectories(Tsb_est, Tsb_gt, units='m'):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(Tsb_est[0,:], Tsb_est[1,:], Tsb_est[2,:], c='r')
  ax.plot(Tsb_gt[0,:], Tsb_gt[1,:], Tsb_gt[2,:], c='b')
  ax.set_xlabel('x ({})'.format(units))
  ax.set_ylabel('y ({})'.format(units))
  ax.set_zlabel('z ({})'.format(units))
  ax.legend(('estimated', 'ground truth'))


def inlier_outlier_time_three_plots(time_axis, signals, suptitle, inlier_idx,
  outlier_idx, titles=None, xlabel=None):
  if titles is None:
    titles = [None, None, None]
  
  plt.figure()
  plt.suptitle(suptitle)
  plt.subplot(3,1,1)
  _inlier_oulier_time_scatter(time_axis, signals[0,:], inlier_idx, outlier_idx,
    title=titles[0], ylabel="x-axis", show_legend=True)
  plt.subplot(3,1,2)
  _inlier_oulier_time_scatter(time_axis, signals[1,:], inlier_idx, outlier_idx,
    title=titles[1], ylabel="y-axis", show_legend=False)
  plt.subplot(3,1,3)
  _inlier_oulier_time_scatter(time_axis, signals[2,:], inlier_idx, outlier_idx,
    title=titles[2], ylabel="z-axis", show_legend=False, xlabel=xlabel)


def inlier_outlier_time_scatter(time_axis, signal, inlier_idx, outlier_idx,
  title=None, ylabel=None, xlabel=None, show_legend=True):
  plt.figure()
  _inlier_oulier_time_scatter(time_axis, signal, inlier_idx, outlier_idx,
  title=title, ylabel=ylabel, xlabel=xlabel, show_legend=show_legend)


def _inlier_oulier_time_scatter(time_axis, signal, inlier_idx, outlier_idx,
  title=None, ylabel=None, xlabel=None, show_legend=True):
  plt.scatter(time_axis[inlier_idx], signal[inlier_idx], marker='.')
  plt.scatter(time_axis[outlier_idx], signal[outlier_idx], marker='.')
  if show_legend:
    plt.legend(('inliers', 'outliers'))
  if title is not None:
    plt.title(title)
  if ylabel is not None:
    plt.ylabel(ylabel)
  if xlabel is not None:
    plt.xlabel(xlabel)


def inlier_outlier_error_three_plots(time_axis, error_signals, inlier_idx,
  outlier_idx, seq, error_type, error_unit):
  means = np.mean(error_signals, axis=1)
  var = np.var(error_signals, axis=1)
  print("{} {} error means: {} {}".format(seq, error_type, means, error_unit))
  print("{} {} error variance: {}".format(seq, error_type, var))

  inlier_mean = np.mean(error_signals[:,inlier_idx], axis=1)
  inlier_var = np.var(error_signals[:,inlier_idx], axis=1)
  outlier_mean = np.mean(error_signals[:,outlier_idx], axis=1)
  outlier_var = np.var(error_signals[:,outlier_idx], axis=1)
  print("{} inlier {} error means: {} {}".format(seq, error_type,
    inlier_mean, error_unit))
  print("{} inlier {} error variance: {}".format(seq, error_type, inlier_var))
  print("{} outlier {} error means: {} {}".format(seq, error_type,
    outlier_mean, error_unit))
  print("{} outlier {} error variance: {}".format(seq, error_type, outlier_var))


def inlier_outlier_3D_error_cloud(error_signals, inlier_idx, outlier_idx,
  title):
  fig = plt.figure()
  plt.suptitle(title)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(error_signals[0,inlier_idx], error_signals[1,inlier_idx],
    error_signals[2,inlier_idx], marker='.')
  ax.scatter(error_signals[0,outlier_idx], error_signals[1,outlier_idx],
    error_signals[2,outlier_idx], marker='.')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.legend(('inliers', 'outliers'))


def inlier_outlier_sigma_hist(sigmas, title, inlier_idx, outlier_idx,
  plot_nbins=10):
  plt.figure()
  plt.title(title)
  plt.xlabel(r"# $\sigma$")
  plt.ylabel("Timesteps in Bin")
  plt.hist(sigmas[inlier_idx], bins=plot_nbins, alpha=0.5, edgecolor='black',
    label='inliers')
  plt.hist(sigmas[outlier_idx], bins=plot_nbins, alpha=0.5, edgecolor='black',
    label='outliers')
  plt.legend(loc='upper right')
