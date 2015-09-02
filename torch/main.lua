require 'torch'
require 'nn'

require 'load_dataset'

-- Seed the RNG
torch.manualSeed(12345)

-- Define dataset constants
n_train_samples = 60000
n_test_samples = 10000
geometry = {32, 32}
sample_size = 32*32
-- Define network hyperparameters
hidden1_size = 1024
num_capsules = 10
num_params = 3 -- 1 for probability, 2 for translation coordinates

-- Load train and test sets
train = mnist.loadTrainSet(n_training_samples, geometry)
test = mnist.loadTestSet(n_test_samples, geometry)
-- Normalize train and test sets
train:normalizeGlobal()
test:normalizeGlobal()

-- Define the model
-- autoencoder = nn.Sequential()

-- First define the encoder
encoder = nn.Sequential()
encoder:add(nn.Reshape(sample_size))
encoder:add(nn.Linear(sample_size, hidden1_size))
encoder:add(nn.ReLU())
encoder:add(nn.Linear(hidden1_size, num_capsules*num_params)) -- 'num_params' parameters per capsule for a total of 'num_capsules*num_params' instantiation parameters
encoder:add(nn.ReLU()) -- Debatable. Using a ReLU non-linearity for probability and instantiation parameters is pby not ideal

decoder = nn.ParallelTable()