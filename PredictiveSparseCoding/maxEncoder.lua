local maxEncoder, parent = torch.class('nn.maxEncoder', 'nn.Module')

function maxEncoder:__init(inputSize)
   parent.__init(self)
end

function maxEncoder:updateOutput(input)
   self.output:resizeAs(input);
   self.output:copy(input);
   self.output[torch.lt(input,0)] = 0;
   return self.output;
end

function maxEncoder:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input);
   self.gradInput:fill(1);
   self.gradInput[torch.lt(input,0)] = 0;
   return self.gradInput;
end

