
dofile  'fista.lua'
dofile  'fistaL1.lua'
dofile  'mnist.lua'
dofile  'sparseCode.lua'

function doL1SparseCoding()

  --Size of the sparse representaiton
  dict_size = 256;
  --Learning Rate of W
  dict_lr = 0.1;
  --Get the Data
  train_data = mnist:getDatasets(30000);
  --Direct Solution to W or not
  direct_sol = true;
  --Line Search(TRUE) or Spectral Approximation(FALSE)
  lineSearch = false;
   
  local sparseObj = sparseCoding( train_data[1][1]:size()[1] , dict_size);
  
  local dictionary = sparseObj:learn(train_data,  dict_lr, direct_sol, lineSearch);

  sparseObj.display(); 

end

doL1SparseCoding();



