import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from pomegranate.hmm import DenseHMM
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pomegranate.distributions import Normal
import torch
import cvxpy as cp
import time