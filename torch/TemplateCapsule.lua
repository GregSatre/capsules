require 'nn'

local TemplateCapsule, Parent = torch.class('nn.TemplateCapsule', 'nn.Module')

-- TemplateCapsule constructor
-- inputs:
--          geometry : {int, int}, the geometry of the template
function TemplateCapsule:__init(geometry, output_width)
   Parent.__init(self)
   self.template = torch.randn(geometry[1], geometry[2])
   self.gradTemplate = torch.Tensor(geometry[1], geometry[2])
   self.output = torch.zeros(output_width, output_width)
end

-- input: tensor of size 3
--        input[1] and input[2] are translation parameters
--        input[3] is the probability coefficient
-- output:
--        32*32 translated and probability-weighted template
function TemplateCapsule:updateOutput(input)
    trans_x = input[1]
    trans_y = input[2]
    prob = input[3]

    -- temp_height = self.template:size()[1]
    -- temp_width = self.template:size()[2]
    -- loop over all output image pixels
    for out_x = 1, self.output:size()[1] do
        for out_y = 1, self.output:size()[2] do
            -- pixel location before translation
            in_x = out_x - trans_x
            in_y = out_y - trans_y
            -- First component is y-axis, then x-axis (first row, then column)
            self.output[{out_y, out_x}] = self:getInterpolatedTemplateValue(in_x, in_y)
        end
    end
    self.output:mul(prob)
    return self.output
end

function TemplateCapsule:getInterpolatedTemplateValue(x, y)
    -- Get four surrounding pixel coordinates
    low_x = torch.floor(x)
    high_x = low_x + 1
    low_y = torch.floor(y)
    high_y = low_y + 1

    -- First is row number (y, height), second is column (x, width)
    temp11 = self:getTemplateValue(low_y, low_x, temp_height, temp_width)
    temp12 = self:getTemplateValue(high_y, low_x, temp_height, temp_width)
    temp21 = self:getTemplateValue(low_y, high_x, temp_height, temp_width)
    temp22 = self:getTemplateValue(high_y, high_x, temp_height, temp_width)

    x_remainder = x - low_x
    y_remainder = y - low_y

    tempX1 = x_remainder*temp22 + (1 - x_remainder)*temp12
    tempX2 = x_remainder*temp21 + (1 - x_remainder)*temp11
    tempXY = y_remainder*tempX1 + (1 - y_remainder)*tempX2
    -- Normally divide by (x2 - x1)(y2 - y1) but here this is equal to 1
    return tempXY
end

function TemplateCapsule:getTemplateValue(row, column)
    temp_height = self.template:size()[1]
    temp_width = self.template:size()[2]
    if column < 1 or column > temp_width or row < 1 or row > temp_height then
        return 0
    else
        return self.template[row][column]
    end
end

function TemplateCapsule:updateGradInput(input, gradOutput)
    -- input is the input to the module
    -- gradOutput is the gradient with respect to the output of the module
    print(gradOutput)
    print(gradOutput:size())

    trans_x = input[1]
    trans_y = input[2]
    prob = input[3]

    grad_out_height = gradOutput:size()[1]
    grad_out_width = gradOutput:size()[2]

    grad_t_x = torch.zeros(grad_out_height, grad_out_width)
    grad_t_y = torch.zeros(grad_out_height, grad_out_width)
    grad_prob = torch.zeros(grad_out_height, grad_out_width)

    if self.gradInput then
        self.gradInput:resize(input:size()[1], grad_out_height, grad_out_width):zero()
        print(self.gradInput)
        -- calculate the gradients
        for row = 1, grad_out_height do
            for column = 1, grad_out_width do
                -- pixel location before translation
                in_x = column - trans_x
                in_y = row - trans_y

                -- Get four surrounding pixel coordinates
                low_x = torch.floor(in_x)
                high_x = low_x + 1
                low_y = torch.floor(in_y)
                high_y = low_y + 1

                -- First is row number (y, height), second is column (x, width)
                temp11 = self:getTemplateValue(low_y, low_x, temp_height, temp_width)
                temp12 = self:getTemplateValue(high_y, low_x, temp_height, temp_width)
                temp21 = self:getTemplateValue(low_y, high_x, temp_height, temp_width)
                temp22 = self:getTemplateValue(high_y, high_x, temp_height, temp_width)

                x_remainder = in_x - low_x
                y_remainder = in_y - low_y

                tempX1 = x_remainder*temp22 + (1 - x_remainder)*temp12
                tempX2 = x_remainder*temp21 + (1 - x_remainder)*temp11
                tempXY = y_remainder*tempX1 + (1 - y_remainder)*tempX2

                -- First component is y-axis, then x-axis (first row, then column)
                grad_t_x[{row, column}] = y_remainder*(temp22 - temp12) + (1 - y_remainder)*(temp21 - temp11)
                grad_t_x[{row, column}] = x_remainder*(temp22 - temp21) + (1 - x_remainder)*(temp12 - temp11)
                grad_prob[{row, column}] = tempXY
            end
        end
        -- scale by (-prob) to complete the gradients
        grad_t_x:mul(-prob)
        grad_t_y:mul(-prob)

        self.gradInput[1] = grad_t_x
        self.gradInput[2] = grad_t_y
        self.gradInput[3] = grad_prob

        print(self.gradInput)

        return self.gradInput
    end
end

function TemplateCapsule:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    -- Gradient of output with regards to parameters is simply 1
    self.gradTemplate:add(scale, gradOutput)
end

function TemplateCapsule:reset()
    self.output = torch.zeros(self.output:size()[1], self.output:size()[1])
end