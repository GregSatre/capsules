local IntensityMod, parent = torch.class('nn.IntensityMod', 'nn.Module')

function IntensityMod:__init()
    parent.__init(self)
end

-- input format : {intensities, images}
-- intensities are Batch*NumCapsules
-- images are Batch*NumCapsules*Height*Width
function IntensityMod:updateOutput(input)
    -- output is Batch*NumCapsules*Height*Width, just like images
    self.output = input[2]:clone()
    -- for i=1,input[1]:size(1) do
    --     self.output[i] = self.output[i] * input[1][i][1]
    -- end
    local batchSize = input[1]:size(1)
    local numCapsules = input[1]:size(2)
    local intensities = input[1]:view(batchSize, numCapsules, 1, 1):expand(input[2]:size())
    self.output:cmul(intensities)
    return self.output
end

function IntensityMod:updateGradInput(input, gradOutput)
    self.gradInput = {torch.Tensor(input[1]:size()):zero(), torch.Tensor(input[2]:size()):zero()}
    -- for i=1,input[1]:size(1) do
    --     self.gradInput[1][i] = torch.cmul(input[2][i], gradOutput[i]):sum()
    --     self.gradInput[2][i] = gradOutput[i] * input[1][i][1]
    -- end
    local batchSize = input[1]:size(1)
    local numCapsules = input[1]:size(2)
    local intensities = input[1]:view(batchSize, numCapsules, 1, 1):expand(input[2]:size())


    self.gradInput[2] = torch.cmul(gradOutput, intensities)
    return self.gradInput
end