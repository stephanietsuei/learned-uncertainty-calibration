import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from utils import calc_avg_sampling_freq, get_stock_parser


parser = get_stock_parser("Plots timeseries and FFTs of IMU data")


class IMUData:
  def __init__(self, imu_data_file):
    self.data_filepath = imu_data_file
    self.timestamps = []
    self.data = []
    self.read_data()
    self.sampling_freq = calc_avg_sampling_freq(self.timestamps)


  def read_data(self):
    with open(self.data_filepath, 'r') as fid:
      for line in fid.readlines():

        # skip the first line
        if line[0] == '#':
          continue

        larr = line.split(',')
        self.timestamps.append(float(larr[0]))
        self.data.append([float(larr[1]), float(larr[2]), float(larr[3]),
                          float(larr[4]), float(larr[5]), float(larr[6])])

    self.timestamps = np.array(self.timestamps)
    self.timestamps = self.timestamps - self.timestamps[0]
    self.timestamps = self.timestamps * 1e-9

    self.data = np.array(self.data)


  def plot_timeseries(self, cols, title, ylabels):
    num_plots = len(cols)

    plt.figure()
    plt.suptitle(title)
    for ind,col in enumerate(cols):
      plt.subplot(num_plots,1,ind+1)
      plt.plot(self.timestamps, self.data[:,col])
      plt.ylabel(ylabels[ind])
    plt.xlabel('Time (s)')


  def plot_fft(self, cols, title, ylabels, max_freq_Hz=None):
    num_plots = len(cols)

    plt.figure()
    plt.suptitle(title)
    for ind,col in enumerate(cols):
      # compute fft
      fft = np.abs(np.fft.rfft(self.data[:,col]))
      freqs = np.fft.rfftfreq(len(self.timestamps), d=1/self.sampling_freq)

      # make the plot
      plt.subplot(num_plots,1,ind+1)
      plt.plot(freqs, fft)
      plt.ylabel(ylabels[ind])
      plt.ylim(bottom=0)

      if max_freq_Hz is not None:
        plt.xlim([0, max_freq_Hz])
    plt.xlabel('Frequency (Hz)')


  def plot_accel(self, max_freq_Hz=None):
    self.plot_timeseries([3,4,5], r'Accelerometer Measurements (m/$s^2$)',
      ['x', 'y', 'z'])
    self.plot_fft([3,4,5], 'Accelerometer FFT', ['x', 'y', 'z'],
      max_freq_Hz=max_freq_Hz)


  def plot_gyro(self, max_freq_Hz=None):
    self.plot_timeseries([0,1,2], 'Gyroscope Measurements (rad/s)',
      ['x', 'y', 'z'])
    self.plot_fft([0,1,2], 'Gyroscope FFT', ['x', 'y', 'z'],
      max_freq_Hz=max_freq_Hz)



if __name__=="__main__":
  args = parser.parse_args()
  if args.dataset == 'tumvi':
    imu_data_file = os.path.join(args.root,
      'dataset-{}_512_16'.format(args.seq), 'mav0', 'imu0', 'data.csv')
    max_freq_Hz = 4.0
  elif args.dataset == 'cosyvio':
    imu_data_file = os.path.join(args.root, 'data', args.sen, args.seq,
      'data.csv')
    max_freq_Hz = 10.0
  elif args.dataset == 'carla':
    imu_data_file = os.path.join(args.root, args.seq, 'imu', 'data.csv')
    max_freq_Hz = 4.0
  elif args.dataset in ['alphred', 'sabr']:
    imu_data_file = os.path.join(args.root, args.seq, 'imu0', 'data.csv')
    max_freq_Hz = 30.0
  elif args.dataset == 'xivo':
    imu_data_file = os.path.join(args.root, args.seq, 'imu0', 'data.csv')
    max_freq_Hz = 30.0

  imu = IMUData(imu_data_file)
  imu.plot_accel(max_freq_Hz=max_freq_Hz)
  imu.plot_gyro(max_freq_Hz=max_freq_Hz)
  plt.show()
