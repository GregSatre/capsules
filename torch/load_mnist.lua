require 'torch'
require 'image'

train = torch.load('mnist.t7/train_32x32.t7', 'ascii')
test = torch.load('mnist.t7/test_32x32.t7', 'ascii')

print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().
-- print('training data:')
-- image.display(train.data[{ {1,256} }])
-- print('test data:')
-- image.display(test.data[{ {1,256} }])


-- Change tensor type. When loaded it is ByteTensor but mean() is not implemented for ByteTensors
train.data = train.data:type(torch.getdefaulttensortype())
test.data = test.data:type(torch.getdefaulttensortype())

-- Get data between 0 and 1
-- max = train.data:max()
-- train.data:div(max)
-- test.data:div(max)

-- Normalize train data
mean = train.data:mean()
std = train.data:std()

train.data:add(-mean)
train.data:div(std)

-- Normalize test data with training mean and std dev

test.data:add(-mean)
test.data:div(std)