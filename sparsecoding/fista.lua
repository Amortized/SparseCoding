function FistaLS(f, g, pl, xinit, lineSearch)

   local params ={}
   local L = 0.1
   local Lstep = 1.5
   local Lrate = 0.2
   local maxiter = 50
   local maxline = 20
   local errthres = 1e-4
   local verbose = false


   -- temporary allocations
   params.xkm = params.xkm or torch.Tensor()
   params.y   = params.y   or torch.Tensor()
   params.ply = params.ply or torch.Tensor()
   local xkm = params.xkm  -- previous iteration
   local y   = params.y    -- fista iteration
   local ply = params.ply  -- soft shrinked y

   -- we start from all zeros
   local xk = xinit
   xkm:resizeAs(xk):zero()
   ply:resizeAs(xk):zero()
   y:resizeAs(xk):zero()

   local history = {} -- keep track of stuff
   local niter = 0    -- number of iterations done
   local converged = false  -- are we done?
   local tk = 1      -- momentum param for FISTA
   local tkp = 0


   local gy = g(y)
   local fval = math.huge -- fval = f+g
   while not converged and niter < maxiter do

      -- run through smooth function (code is input, input is target)
      -- get derivatives from smooth function
    local fy,gfy = f(y,'dx')
  
    --Linear search or spectral approximation 
    if niter == 0 or lineSearch == true then
	
	  --print("LinearSearch")
	  --------------------------- LINEAR SEARCH ----------------------------
    	  local fply = 0
          local gply = 0
          local Q = 0
          local nline = 0
          local linesearchdone = false

         while not linesearchdone do
   	      -- take a step in gradient direction of smooth function
        	 local ply = y:clone()
	         ply:add(-1/L,gfy)

              -- and solve for minimum of auxiliary problem
	         pl(ply,L)
              -- this is candidate for new current iteration
                 xk:copy(ply)
              -- evaluate this point F(ply)
                 fply = f(ply)
              -- ply - y
	         ply:add(-1, y)
              -- <ply-y , \Grad(f(y))>
	         local Q2 = gfy:dot(ply)
              -- L/2 ||beta-y||^2
	         local Q3 = L/2 * ply:dot(ply)
              -- Q(beta,y) = F(y) + <beta-y , \Grad(F(y))> + L/2||beta-y||^2 + G(beta)
	         Q = fy + Q2 + Q3
	      -- check if F(beta) < Q(pl(y),\t)
        	if fply <= Q then --and Fply + Gply <= F then
            	  -- now evaluate G here  
	          linesearchdone = true
	        elseif  nline >= maxline then
	          linesearchdone = true
	          xk:copy(xkm) -- if we can't find a better point, current iter = previous iter
         	else
            	   L = L * Lstep
         	end
        	nline = nline + 1
      	 end
    else 
	 --------------------------- SPECTRAL APPROXIMATION ----------------------------
	  --print("Spectral Approx");
          local specApproxSearchDone = false
	  local spLine = 0;
	  local v_k = torch.rand(y:size());
	  local L_n = torch.norm(v_k);
	  local L_p = torch.norm(v_k);
	  local beta = 0.2;
	  gfy_p = gfy:clone();
	  while not specApproxSearchDone do		
	    --Compute v_k-1/ || v_k - 1 ||
	    v_k:div(L_n);
	    fy, gfy = f(y + v_k, 'dx');
	    --v_k is diff between the gradients
	    v_k = gfy - gfy_p;
	    L_n = torch.norm(v_k);
	    --Check for convergnce
	    if torch.abs(L_n - L_p) < 1e-10 or spLine >= maxline then 
		--Now evalulate the Lipschitz constant using L formed from spec approx
		L = ((beta * L) + (1 - beta) * L_n);
	        --Figure out the candidate at this step of iteration
		fy, gfy = f(y, 'dx');
		local ply = y:clone()
                ply:add(-1/L,gfy)
                pl(ply,L)
                xk:copy(ply)
	        --Set to break out
		specApproxSearchDone = true;
	    else
		L_p = L_n;
		spLine = spLine + 1;		
	    end	
     	 	
	  end
   end
      ---------------------------------------------
      -- FISTA
      ---------------------------------------------
         -- do the FISTA step
         tkp = (1 + math.sqrt(1 + 4*tk*tk)) / 2
         -- x(k-1) = x(k-1) - x(k)
         xkm:add(-1,xk)
         -- y(k+1) = x(k) + (1-t(k)/t(k+1))*(x(k-1)-x(k))
         y:copy(xk)
         y:add( (1-tk)/tkp , xkm)
         -- store for next iterations
         -- x(k-1) = x(k)
         xkm:copy(xk)

      -- t(k) = t(k+1)
      tk = tkp
      local fply = f(y)
      local gply = g(y)
      if verbose then
	 print(string.format('iter=%d eold=%g enew=%g',niter,fval,fply+gply))
      end

      niter = niter + 1

      -- bookeeping
      fval = fply + gply
      history[niter] = {}
      history[niter].nline = nline
      history[niter].L  = L
      history[niter].F  = fval
      history[niter].Fply = fply
      history[niter].Gply = gply
      params.L = L
      if verbose then
         history[niter].xk = xk:clone()
         history[niter].y  = y:clone()
      end

      -- are we done?
      if niter > 1 and math.abs(history[niter].F - history[niter-1].F) <= errthres then
         converged = true
	 xinit:copy(y)
         return y,history
      end

      if niter >= maxiter then
	 xinit:copy(y)
         return y,history
      end

   end
   error('not supposed to be here')
end

