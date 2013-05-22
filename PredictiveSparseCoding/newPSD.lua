dofile 'maxEncoder.lua';

local newPSD, parent = torch.class('unsup.newPSD','unsup.PSD');

-- inputSize   : size of input
-- outputSize  : size of code
-- lambda      : sparsity coefficient
-- beta	       : prediction coefficient
-- params      : optim.FistaLS parameters
function newPSD:__init(inputSize, outputSize, lambda, beta, params)
   
   -- prediction weight
   self.beta = beta

   -- decoder is L1 solution
   self.decoder = unsup.LinearFistaL1(inputSize, outputSize, lambda, params)

   -- encoder
   params = params or {}
   self.params = params
   self.params.encoderType = 'maxEncoder' or 'softPlus';

   if params.encoderType == 'softPlus' then
      self.encoder = nn.Sequential();
      self.encoder:add(nn.Linear(inputSize,outputSize));
      self.encoder:add(nn.SoftPlus());
   elseif params.encoderType == 'maxEncoder' then
      self.encoder = nn.Sequential();
      self.encoder:add(nn.Linear(inputSize,outputSize));
      self.encoder:add(nn.maxEncoder());
   else
      error('params.encoderType unknown " ' .. params.encoderType)
   end

   parent.__init(self, self.encoder, self.decoder, self.beta, self.params)

end
