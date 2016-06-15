require 'nn'

local TemplateConstant, Parent = torch.class('nn.TemplateConstant', 'nn.Module')

function TemplateConstant:__init(size, height, width)
    Parent.__init(self)
    self.templates = torch.Tensor(size, height, width)
    self.gradTemplates = torch.Tensor(size, height, width)
end

-- input format : { (batchSize, sizeIn, 1, height, width) templates Tensor,
--                  (batchSize, sizeIn, sizeOut, 1,1) intensities Tensor}
-- output format : (batchSize, sizeIn, sizeOut, height, width) Tensor
function TemplateConstant:updateOutput(input)
    self.output = self.templates:copy()
    return self.output
end

function TemplateConstant:updateGradInput(input, gradOutput)
    self.gradInput[2] = gradOutput
    return self.gradInput
end

function TemplateConstant:accGradParameters(input, gradOutput, scale)
    self.gradTemplates:add(scale, gradOutput)
end

function TemplateConstant:zeroGradParameters()
    self.gradTemplates:zero()
end

function TemplateConstant:parameters()
    return {self.templates}, {self.gradTemplates}
end

function TemplateConstant:reset()
    self.templates = torch.randn(self.templates:size())
    self.gradTemplates:zero()
end
