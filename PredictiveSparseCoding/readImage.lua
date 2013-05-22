require 'torch'
require 'image'

local imageMap = torch.class('imageMap');

function imageMap:__init(train_size)

   --Data
   self.data = {};

   --Read the list of files in the directory
   local file = 'listFiles.dat'; 
   local fp = assert(io.open (file));
   local lines = {};
   local tr_size = train_size

   --Store the File Names
   fc = 1;
   for line in fp:lines() do
      lines[fc] = line;
      fc = fc + 1;
   end
   --Generate a random order
   local rorder = torch.randperm(train_size);
   --Prepare a Map from key -- filename
   local fileMap = {};
   for i = 1, train_size do
     --Randomly pick an image
     fileMap[i] = lines[rorder[i]];
   end
  
  --Overloading the index operator 
  setmetatable(self.data, {
  			      __index = function(t, key)
			      local im = image.load('./train256/'..fileMap[key%tr_size+1]);
			      --If a single channel image
			      if(im:size(1)==1) then
		                local tmp = torch.Tensor(3,im:size(2),im:size(3));
		                tmp[1]=im[1];
                		tmp[2]=im[1];
		                tmp[3]=im[1];
                		im=tmp;
			      end
			      im = image.rgb2yuv(im)
			      -- Return the 1st dimension of image
			      return im;
    			      end
    			})

   
end

function getChannel(image, channel)
   return image[channel];
end


function getPatches(image, size, kernelSize)
        patches = {}
        top_patches = {};
        --Compute the Size of the Patch for a given KernelSize
        size = size + 2*torch.floor(kernelSize/2)
        --No of Patches
        local no = 0;
        --Image has to be already channeled
        local image = image;
        local height = image:size(1);
        local width = image:size(2);

	--No of Patches
        function top_patches:size() return 50 end;
	
        --Randomly permuting the patches
        local rorder = torch.randperm(  torch.floor(width/size)*torch.floor(height/size)  );

        for i = 1,height-size+1,size do
                for j = 1,width-size+1,size do
                        no = no + 1
                        patches[rorder[no]] = image[{{i,i+size-1},{j,j+size-1}}];
                        patches[rorder[no]] = lcn(patches[rorder[no]],kernelSize):clone();
                        patches[rorder[no]]:resize((size-4)*(size-4));
                end
        end

	--Return the permutted 100 patches
	count = 1;
        for i = 1, rorder:size()[1] do
	   top_patches[i] = patches[rorder[i]];
	   if count == 50 then 
	     break;	 	
	   end
	   count = count + 1;
	end

        return top_patches;
end


--Function Adapted from demo_data in unsup package......
function lcn(im,kernelsize)
      local gs = kernelsize
      local imsq = torch.Tensor()
      local lmnh = torch.Tensor()
      local lmn = torch.Tensor()
      local lmnsqh = torch.Tensor()
      local lmnsq = torch.Tensor()
      local lvar = torch.Tensor()
      local gfh = image.gaussian{width=gs,height=1,normalize=true}
      local gfv = image.gaussian{width=1,height=gs,normalize=true}
      local gf = image.gaussian{width=gs,height=gs,normalize=true}

      local mn = im:mean()
      local std = im:std()
      if data_verbose then
         print('im',mn,std,im:min(),im:max())
      end
      im:add(-mn)
      im:div(std+1e-12)
      if data_verbose then
         print('im',im:min(),im:max(),im:mean(), im:std())
      end

      imsq:resizeAs(im):copy(im):cmul(im)
      if data_verbose then
         print('imsq',imsq:min(),imsq:max())
      end

      lmnh=torch.conv2(im,gfh)
      lmn=torch.conv2(lmnh,gfv)
      if data_verbose then
         print('lmn',lmn:min(),lmn:max())
      end

      --local lmn = torch.conv2(im,gf)
      torch.conv2(lmnsqh,imsq,gfh)
      torch.conv2(lmnsq,lmnsqh,gfv)
      if data_verbose then
         print('lmnsq',lmnsq:min(),lmnsq:max())
      end

      lvar:resizeAs(lmn):copy(lmn):cmul(lmn)
      lvar:mul(-1)
      lvar:add(lmnsq)
      if data_verbose then
         print('2',lvar:min(),lvar:max())
      end

      --lvar:apply(function (x) if x<0 then return 0 else return x end end)
      lvar[torch.lt(lvar,0)] = 0
      if data_verbose then
         print('2',lvar:min(),lvar:max())
      end


      local lstd = lvar
      lstd:sqrt()
	
      --lstd:apply(function (x) if x<1 then return 1 else return x end end)
      lstd[torch.lt(lstd,1)]=1

      if data_verbose then
         print('lstd',lstd:min(),lstd:max())
      end

      local shift = (gs+1)/2
      local nim = im:narrow(1,shift,im:size(1)-(gs-1)):narrow(2,shift,im:size(2)-(gs-1))
      nim:add(-1,lmn)
      nim:cdiv(lstd)
      if data_verbose then
         print('nim',nim:min(),nim:max())
      end
      return nim
end

--data = imageMap(10).data;

--print(data[1]);
