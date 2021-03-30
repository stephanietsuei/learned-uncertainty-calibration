import sys, os
import numpy as np
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from pltutils import interval_bounds_plot
from utils import chi2_divergence, compute_scaled_eigvecs


class DiscreteTimeSystemSimulation:
  def __init__(self, system, estimator, x0, P0, meas_dim, inputs, nruns):
    self.system = system
    self.estimator = estimator
    self.inputs = inputs
    self.nruns = nruns
    self.num_timesteps = inputs.shape[0]
    self.meas_dim = meas_dim

    # data to save during simulations
    self.state_dim = x0.shape[0]
    self.x = np.zeros((self.nruns, self.num_timesteps, self.state_dim))
    self.xest = np.zeros((self.nruns, self.num_timesteps, self.state_dim))
    self.x_mean = np.zeros((self.num_timesteps, self.state_dim))
    self.xest_mean = np.zeros((self.num_timesteps, self.state_dim))
    self.Pest = np.zeros((self.nruns, self.num_timesteps,
                          self.state_dim, self.state_dim))
    self.inn = np.zeros((self.nruns, self.num_timesteps, self.meas_dim))
    self.inn_mean = np.zeros((self.num_timesteps, self.meas_dim))

    # Analysis data (NEES and true Monte-Carlo covariance)
    self.P = np.zeros((self.num_timesteps, self.state_dim, self.state_dim))
    self.NEES_est = np.zeros((self.nruns, self.num_timesteps))
    self.NEES_true = np.zeros((self.nruns, self.num_timesteps))
    self.divergence_est = np.zeros((self.num_timesteps,))
    self.divergence_true = np.zeros((self.num_timesteps,))
    self.divergence_runs_est = np.zeros((self.nruns,))
    self.divergence_runs_true = np.zeros((self.nruns,))

    # covariance quantification, computed using self.Pest and self.P, respectively
    self.eigvec_coord_est = np.zeros((self.nruns, self.num_timesteps, self.state_dim))
    self.eigvec_coord_true = np.zeros((self.nruns, self.num_timesteps, self.state_dim))
    self.evec_scales_est = np.zeros((self.nruns, self.num_timesteps, self.state_dim))
    self.evec_scales_true = np.zeros((self.nruns, self.num_timesteps, self.state_dim))
    self.sigma_est = np.zeros((self.nruns, self.num_timesteps))
    self.sigma_true = np.zeros((self.nruns, self.num_timesteps))


  def sim_one_run(self, run_idx):
    for i in range(self.num_timesteps):
      # record data from last timestep
      self.x[run_idx,i,:] = self.system.get_true_state()
      self.xest[run_idx,i,:] = self.estimator.get_est()
      self.Pest[run_idx,i,:,:] = self.estimator.get_cov()
      self.inn[run_idx,i,:] = self.estimator.get_inn()

      # propagate real system to next timestep
      self.system.propagate(self.inputs[i,:])
      y = self.system.meas()

      # filter update for next timestep
      self.estimator.predict(self.inputs[i,:])
      self.estimator.measurement_update(y)


  def sim_all_runs(self):
    for i in range(self.nruns):
      self.system.reset()
      self.estimator.reset()
      self.sim_one_run(i)

  
  def calc_mean(self):
    self.x_mean = np.mean(self.x, axis=0)
    self.xest_mean = np.mean(self.xest, axis=0)
    self.inn_mean = np.mean(self.inn, axis=0)
  

  def calc_true_cov(self):
    for i in range(self.num_timesteps):
      total_P = np.zeros((self.state_dim, self.state_dim))
      for j in range(self.nruns):
        diff = self.xest[j,i,:] - self.x[j,i,:]
        total_P += np.outer(diff, diff)
      self.P[i,:,:] = total_P / (self.nruns - 1)
  

  def calc_tot_NEES(self):
    self.singular_indices = [ set() for i in range(self.nruns) ]
    for j in range(self.nruns):
      for i in range(self.num_timesteps):
        diff = self.xest[j,i,:] - self.x[j,i,:]
        Pest = self.Pest[j,i,:,:]
        Ptrue = self.P[i,:,:]

        try:
          (Xest, evec_scales_est) = \
            compute_scaled_eigvecs(Pest, return_scales=True)
          (Xtrue, evec_scales_true) = \
            compute_scaled_eigvecs(Ptrue, return_scales=True)
          CoordInEigvecsEst = np.linalg.solve(Xest, diff)
          CoordInEigvecsTrue = np.linalg.solve(Xtrue, diff)
          self.sigma_est[j,i] = np.linalg.norm(CoordInEigvecsEst)
          self.sigma_true[j,i] = np.linalg.norm(CoordInEigvecsTrue)
          self.eigvec_coord_est[j,i,:] = CoordInEigvecsEst
          self.eigvec_coord_true[j,i,:] = CoordInEigvecsTrue
          self.evec_scales_est[j,i,:] = evec_scales_est
          self.evec_scales_true[j,i,:] = evec_scales_true
          self.NEES_est[j,i] = self.sigma_est[j,i]**2
          self.NEES_true[j,i] = self.sigma_true[j,i]**2
        except np.linalg.LinAlgError:
          self.singular_indices[j].add(i)

      if len(self.singular_indices[j]) > 0:
        print("Warning: on run {}, {} indices had singular covariances".format(
          j, len(self.singular_indices[j])
        ))


  def calc_divergence(self, ret1=False, nbins='auto', int_upper_lim=100.0):
    # for each run
    for j in range(self.nruns):
      all_timesteps = set([i for i in range(self.num_timesteps)])
      good_timesteps = list(all_timesteps - self.singular_indices[j])
      self.divergence_runs_est[j] = \
        chi2_divergence(self.NEES_est[j,good_timesteps],
                        self.state_dim,
                        hist_nbins=nbins,
                        int_upper_lim=int_upper_lim)
      self.divergence_runs_true[j] = \
        chi2_divergence(self.NEES_true[j,:],
                        self.state_dim,
                        hist_nbins=nbins,
                        int_upper_lim=int_upper_lim)
    if ret1:
      return (self.divergence_runs_est[0], self.divergence_runs_true[0])
    else:
      average_est_div = np.mean(self.divergence_runs_est)
      est_div_std = np.std(self.divergence_runs_est)
      average_true_div = np.mean(self.divergence_runs_true)
      est_true_std = np.std(self.divergence_runs_true)
      print("Average est div: {} +/- {}".format(average_est_div, est_div_std))
      print("Average true div: {} +/- {}".format(average_true_div, est_true_std))
      return (average_est_div, average_true_div)


  def compute_sigma_percentages(self):
    num_1sigma_byaxis_est = np.zeros((self.state_dim,))
    num_2sigma_byaxis_est = np.zeros((self.state_dim,))
    num_3sigma_byaxis_est = np.zeros((self.state_dim,))

    num_1sigma_byaxis_true = np.zeros((self.state_dim,))
    num_2sigma_byaxis_true = np.zeros((self.state_dim,))
    num_3sigma_byaxis_true = np.zeros((self.state_dim,))

    # Count, by axis evec_scales calculation
    for j in range(self.nruns):
      for i in range(self.num_timesteps):
        for k in range(self.state_dim):
          val = np.abs(self.eigvec_coord_est[j,i,k])
          if val <= 3.0:
            num_3sigma_byaxis_est[k] += 1
          if val <= 2.0:
            num_2sigma_byaxis_est[k] += 1
          if val <= 1.0:
            num_1sigma_byaxis_est[k] += 1

          val = np.abs(self.eigvec_coord_true[j,i,k])
          if val <= 3.0:
            num_3sigma_byaxis_true[k] += 1
          if val <= 2.0:
            num_2sigma_byaxis_true[k] += 1
          if val <= 1.0:
            num_1sigma_byaxis_true[k] += 1
    
    num_1sigma_byaxis_est /= self.nruns * self.num_timesteps
    num_2sigma_byaxis_est /= self.nruns * self.num_timesteps
    num_3sigma_byaxis_est /= self.nruns * self.num_timesteps
    num_1sigma_byaxis_true /= self.nruns * self.num_timesteps
    num_2sigma_byaxis_true /= self.nruns * self.num_timesteps
    num_3sigma_byaxis_true /= self.nruns * self.num_timesteps
   
    # print
    print("Per-Axis Est (1,2,3):")
    for k in range(self.state_dim):
      print("State {}: {}, {}, {}".format(
        k,
        num_1sigma_byaxis_est[k],
        num_2sigma_byaxis_est[k],
        num_3sigma_byaxis_est[k]
      ))

    print("Per-Axis True (1,2,3):")
    for k in range(self.state_dim):
      print("State {}: {}, {}, {}".format(
        k,
        num_1sigma_byaxis_true[k],
        num_2sigma_byaxis_true[k],
        num_3sigma_byaxis_true[k]
      ))


  def plt_timeseries(self, stepsize=20):
    plt.figure()
    tpts = np.arange(0,self.num_timesteps,stepsize)
    for i in range(self.state_dim):
      plt.subplot(self.state_dim, 1, i+1)
      plt.plot(self.x[0,:,i], 'b')
      plt.plot(tpts,self.xest[0,tpts,i], '*r')
      plt.ylabel("x{}".format(i+1))
      #if i == 0:
      #  plt.legend(("estimate", "truth"))
      #if i == self.state_dim-1:
      #  plt.xlabel("Timesteps")
    #plt.suptitle("Timeseries Plot")


  def plt_inn(self, sample_freq, steps_to_chop=10):
    plt.figure()
    for i in range(self.meas_dim):
      plt.subplot(self.meas_dim, 1, i+1)
      plt.plot(self.inn_mean[steps_to_chop:,i], 'b')
      #if i == self.meas_dim-1:
      #  plt.xlabel("Timesteps")
    #plt.suptitle("Mean Innovation Timeseries")

    f_inn, psd_inn = periodogram(self.inn_mean[steps_to_chop:], sample_freq, axis=0)
    psd_inn = np.sqrt(psd_inn)
    plt.figure()
    for i in range(self.meas_dim):
      plt.subplot(self.meas_dim,1,i+1)
      plt.semilogy(f_inn, psd_inn[:,i])
      if i == self.meas_dim-1:
        plt.xlabel("Frequency (Hz)")


  def analysis(self):
    print("simulating...")
    t0 = time.time()
    self.sim_all_runs()
    t1 = time.time()
    print("simulation time: {}".format(t1 - t0))

    print("computing mean trajectory")
    self.calc_mean()
    t2 = time.time()
    print("mean time: {}".format(t2 - t1))

    print("computing true covariance")
    self.calc_true_cov()
    t3 = time.time()
    print("true cov time: {}".format(t3 - t2))

    print("computing NEES")
    self.calc_tot_NEES()
    t4 = time.time()
    print("computing NEES time: {}".format(t4-t3))

    print("computing divergence")
    self.calc_divergence()
    t5 = time.time()
    print("computing divergence time: {}".format(t5-t4))

    self.compute_sigma_percentages()