require 'image'

dofile("fistaL1.lua");


--Does L1 Sparse Coding 

function sparseCoding(feature_size, dict_size)

   --Controller
   sparseObj = {}

   --Dictionary
   sparseObj.n = feature_size;
   sparseObj.m = dict_size;
   sparseObj.dict = torch.randn(sparseObj.n, sparseObj.m)   
     
   fistaParam = {};
   fistaParam.fistaLambda = 0.01

   sparseObj.graddict = torch.Tensor()

   function sparseObj:normalize()
        for i=1, sparseObj.m do
           local colNorm = torch.norm(sparseObj.dict:select(2,i));
           sparseObj.dict:select(2,i):div(colNorm);
        end
   end 
    
   --Subtracts the gradient and normalizes Dict
   function sparseObj:updateDict(input, sparse, cur_reg)
		--Calculate the Gradient
	        reconstruction = torch.Tensor();
 		reconstruction:resize(sparseObj.dict:size(1));
		reconstruction = torch.mv(sparseObj.dict, sparse);
		diff = reconstruction - input
		gradient = torch.zeros(sparseObj.n, sparseObj.m);
		gradient:addr(diff, sparse);
	        --Update the Dict
		sparseObj.dict = sparseObj.dict - (gradient*2*cur_reg);
		--Normalize the Dict
                sparseObj:normalize()
   end

   function sparseObj:learn(data, regularizer, direct_sol, lineSearch)

       params = {};
       params.lineSearch = lineSearch;

       local fista = FistaL1(sparseObj.dict, params.lineSearch)
       local cur_reg = regularizer
       local direct_sol = direct_sol or false;


       --For Direct Sol it is assumed that z is the best representation for x therefore we can figure out 
       --Dict since A * Dict = B where  A = sparse * sparse and B = input * sparse
       local singular_avoid = true;


       A  = torch.zeros(sparseObj.m ,sparseObj.m)
       B = torch.zeros(sparseObj.n,sparseObj.m)

       for s = 1, data:size() do
	  local input = data[s][1];
	  local sparse = fista.run(input, fistaParam.fistaLambda);


	  --Keep Accumulating 
          A:addr(sparse, sparse);
	  B:addr(input, sparse);

	  --Update Dict
	  if direct_sol == false or singular_avoid == true then 
		  sparseObj:updateDict(input, sparse, cur_reg); 
          else
	        --Equate the Dict to the Solution for the current sample	
		--Solve for A * Dict = B
		sparseObj.dict = torch.gesv(B:t(), A:t()):t(); 	
	  end
	  --Change the learning rate
	  if s%100 == 0 then
	      print("Iteration "..s); 
	      cur_reg = (regularizer/(1+s/100));
	      if s%5000 == 0 then
	       --Display the Dict
	       sparseObj:display();
	      end
	      --First 100 runs will make the Dict Non Singular which is required for solving AX = B.
	      singular_avoid = false;
	  end
       end
       return sparseObj.W
	
   end

    
   function sparseObj:display()
                data = {}
                local mi = torch.min(sparseObj.dict)
                local ma = torch.max(sparseObj.dict)
                for i = 1, sparseObj.m do
                        data[i] = sparseObj.dict[{{},i}]:clone():add(-mi):div(ma-mi):pow(6);
                        data[i]:resize(32,32);
                end
	
	       image.display({image=data,nrow=16, min =-0 ,padding=3})
    end


   return sparseObj;

end
