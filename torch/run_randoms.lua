require 'sys'

torch.manualSeed(12345)

function run_experiment(save_dir)
    local lr = tostring(torch.uniform(1e-2,1e-4))
    local weightDecay = torch.uniform(-4,1)
    weightDecay = tostring(torch.exp(weightDecay*torch.log(10)))

    local capSize1 = tostring(torch.round(torch.uniform(4,10)))
    local capSize2 = tostring(torch.round(torch.uniform(6,10)))
    local capSize3 = tostring(torch.round(torch.uniform(6,12)))

    local templateSize = tostring(torch.round(torch.uniform(15,32)))

    local ex_str = 'th test_model.lua -batchSize 50 -optimization RMSProp -learningRate '..lr..' -weightDecay '..weightDecay..' -capSize1 '..capSize1..' -capSize2 '..capSize2..' -capSize3 '..capSize3..' -templateSize '..templateSize..' -clamp -renorm -save '..save_dir
    sys.execute(ex_str)
end

local exp_num = 30
while true do
    exp_num = exp_num+1
    local save_dir = 'results/exp'..tostring(exp_num)
    print("Running experiment "..tostring(exp_num))
    pcall(run_experiment, save_dir)
end
