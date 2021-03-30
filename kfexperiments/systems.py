import numpy as np
from scipy.integrate import solve_ivp



class DiscreteTimeLTI:
  def __init__(self, x0, A, B, C, F, G, R, Q):
    self.A = A
    self.B = B
    self.C = C
    self.F = F  # dynamics noise input
    self.G = G  # measurement noise input
    self.x = x0
    self.x_init = x0
    self.state_dim = self.x.shape[0]
    self.meas_dim = self.C.shape[0]

    # sqrt matrices for noise generation
    # dynamics noise
    evalsR, evecsR = np.linalg.eig(R)
    self.Rsqrt = evecsR @ np.diag(np.sqrt(evalsR))
    # measurement noise
    evalsQ, evecsQ = np.linalg.eig(Q)
    self.Qsqrt = evecsQ @ np.diag(np.sqrt(evalsQ))

  def propagate(self, u):
    # Get propagation noise
    normals = np.random.standard_normal(self.state_dim)
    noise = self.Rsqrt @ normals
    # propagation equation
    self.x = self.A @ self.x + self.B @ u + self.F @ noise

  def meas(self):
    normals = np.random.standard_normal(self.meas_dim)
    noise = self.Qsqrt @ normals
    meas = self.C @ self.x + self.G @ noise
    return meas

  def get_true_state(self):
    return self.x

  def reset(self):
    self.x = self.x_init



def DubinsCarDeriv(x, u):
  v = x[2]
  theta = x[3]
  a = u[0]
  omega = u[1]

  dxdt = np.zeros(x.size)
  dxdt[0] = v*np.cos(theta)
  dxdt[1] = v*np.sin(theta)
  dxdt[2] = a
  dxdt[3] = omega
  return dxdt



def DubinsCarMeas(x, beacon_x, beacon_y):
  num_beacons = len(beacon_x)
  rs = np.zeros(num_beacons)
  phis = np.zeros(num_beacons)
  #robot_pos = x[:2]
  theta = x[3]
  for i in range(num_beacons):
    #beacon_pos = np.array([beacon_x[i], beacon_y[i]])
    #rs[i] = np.linalg.norm(beacon_pos-robot_pos)
    rs[i] = np.sqrt((beacon_x[i]-x[0])**2 + (beacon_y[i]-x[1])**2)

    total_ang = np.arctan2(beacon_y[i]-x[1], beacon_x[i]-x[0])
    phis[i] = total_ang - theta
  meas = np.concatenate((rs, phis))
  return meas




class TwoDLocalizationSystem:
  """a 2D localization problem with a fixed set of beacons. state of the
  robot is (x, y, v, theta). Inputs are (a, omega)"""
  def __init__(self, beacon_x, beacon_y, init_state, dyn_noise_covs,
               meas_noise_covs, dt):

    self.state_dim = 4
    self.num_beacons = len(beacon_x)
    self.meas_dim = 2*self.num_beacons
    self.dt = dt

    self.beacon_x = beacon_x
    self.beacon_y = beacon_y
    self.x = init_state
    self.x_init = init_state

    self.dyn_noise_coef = np.diag(np.sqrt(np.array(dyn_noise_covs)))
    self.meas_noise_coef = np.diag(np.sqrt(np.array(meas_noise_covs)))

  def propagate(self, u):
    """gonna do a small timestep with forward euler integration"""
    rhs = lambda t,s: DubinsCarDeriv(s,u)
    ans = solve_ivp(rhs, (0, self.dt), self.x)
    self.x = ans.y[:,-1]
    self.x += self.dyn_noise_coef @ np.random.standard_normal(self.state_dim)

  def meas(self):
    """meas is (r1, r2, r3, phi1, phi2, phi3)"""
    meas = DubinsCarMeas(self.x, self.beacon_x, self.beacon_y)
    meas += self.meas_noise_coef @ np.random.standard_normal(self.meas_dim)
    return meas

  def get_true_state(self):
    return self.x

  def reset(self):
    self.x = self.x_init



class KalmanFilter:
  """linear kalman filter"""
  def __init__(self, x0, P0, A, B, C, F, G, R, Q):
    self.x_init = x0
    self.P_init = P0
    self.x = x0
    self.P = P0
    self.state_dim = x0.shape[0]
    self.meas_dim = C.shape[0]
    self.inn = np.zeros(self.meas_dim)

    self.A = A
    self.B = B
    self.C = C
    self.F = F
    self.G = G
    self.R = R
    self.Q = Q

  def predict(self, u):
    self.x = self.A @ self.x + self.B @ u
    self.P = self.A @ self.P @ self.A.T + self.F @ self.R @ self.F.T
  
  def measurement_update(self, y):
    y_pred = self.C @ self.x
    inv_part = np.linalg.inv(self.C @ self.P @ self.C.T +
                             self.G @ self.Q @ self.G.T)
    K = self.P @ self.C.T @ inv_part

    self.inn = y - y_pred
    self.x = self.x + K @ self.inn
    ImKC = np.eye(self.state_dim) - K @ self.C
    self.P = ImKC @ self.P @ np.transpose(ImKC) + K @ self.Q @ np.transpose(K)
  
  def get_est(self):
    return self.x

  def get_inn(self):
    return self.inn
  
  def get_cov(self):
    return self.P

  def reset(self):
    self.x = self.x_init
    self.P = self.P_init



def SpringMass(dt, c, k, m, x0, v0, dyn_noise, meas_noise):
  A_cont = np.array([
    [0, 1],
    [-k/m, -c/m]
  ])
  B_cont = np.array([[0], [1/m]])
  A = np.eye(2) + A_cont*dt
  B = B_cont*dt
  C = np.array([[1, 0]])
  F = np.eye(2)
  G = np.eye(1)
  R = np.diag([dyn_noise**2, dyn_noise**2])
  Q = meas_noise**2*np.eye(1)
  x = np.array([x0, v0])
  P0 = np.eye(2)

  x_est_init = x + np.array([0.05, -0.01])

  sys = DiscreteTimeLTI(x, A, B, C, F, G, R, Q)
  KF = KalmanFilter(x+x_est_init, P0, A, B, C, F, G, R, Q)

  return (sys, KF)

class TwoDLocalizationEKF:
  def __init__(self, beacon_x, beacon_y, init_state, init_cov,
               dyn_noise_covs, meas_noise_covs, dt):
    self.state_dim = 4
    self.num_beacons = len(beacon_x)
    self.meas_dim = 2*self.num_beacons
    self.dt = dt

    self.beacon_x = beacon_x
    self.beacon_y = beacon_y
    self.x = init_state
    self.x_init = init_state
    self.P = init_cov
    self.P_init = init_cov
    self.inn = np.zeros(self.meas_dim)

    self.dyn_noise_cov = np.diag(dyn_noise_covs)
    self.meas_noise_cov = np.diag(meas_noise_covs)


  def jacA(self):
    v = self.x[2]
    theta = self.x[3]
    A = np.zeros((self.state_dim, self.state_dim))
    A[0,2] = np.cos(theta)
    A[0,3] = -v * np.sin(theta)
    A[1,2] = np.sin(theta)
    A[1,3] =  v * np.cos(theta)

    Ad = np.eye(self.state_dim) + A*self.dt
    return Ad


  def jacC(self, rs):
    C = np.zeros((self.meas_dim, self.state_dim))
    x = self.x[0]
    y = self.x[1]
    for i in range(self.num_beacons):
      C[2*i,0] = (x - self.beacon_x[i]) / rs[i]
      C[2*i,1] = (y - self.beacon_y[i]) / rs[i]
      C[2*i+1,0] = (self.beacon_y[i] - y) / rs[i] / rs[i]
      C[2*i+1,1] = (x - self.beacon_x[i]) / rs[i] / rs[i]
      C[2*i+1,3] = -1
    return C


  def predict(self, u):
    deriv = lambda t,s: DubinsCarDeriv(s,u)
    ans = solve_ivp(deriv, (0, self.dt), self.x)
    self.x = ans.y[:,-1]

    A = self.jacA()
    self.P = A @ self.P @ A.T + self.dyn_noise_cov


  def measurement_update(self, y):
    y_pred = DubinsCarMeas(self.x, self.beacon_x, self.beacon_y)
    rs = y_pred[:self.num_beacons]
    C = self.jacC(rs)

    inv_part = np.linalg.inv(C @ self.P @ C.T + self.meas_noise_cov)
    K = self.P @ C.T @ inv_part

    self.inn = y - y_pred
    self.x = self.x + K @ self.inn
    self.P = (np.eye(self.state_dim) - K @ C) @ self.P


  def get_est(self):
    return self.x

  def get_inn(self):
    return self.inn

  def get_cov(self):
    return self.P

  def reset(self):
    self.x = self.x_init
    self.P = self.P_init