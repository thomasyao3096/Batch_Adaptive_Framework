import torch
from torch.autograd import Variable
from scipy.stats import norm
import numpy
import math
import time
class batch_adaptive_optimizer(object):
    def __init__(self, model, train_data, train_data_size, valid_data, valid_data_size, prepare_data, criterion, M, max_batch_size, lrate=0.05, gamma=0.9, m0=30, use_diag=True):
        self.model = model
        self.tparams = list(model.parameters())
        self.criterion = criterion
        self.create_Variables()
        self.optimizer = torch.optim.SGD(self.tparams, lr = lrate)
        self.train_data = train_data
        self.train_data_size = train_data_size
        self.valid_data = valid_data
        self.valid_data_size = valid_data_size
        self.prepare_data = prepare_data
        self.M = M
        self.max_batch_size = max_batch_size
        self.gamma = gamma
        self.m0 = m0
        self.use_diag = use_diag
        self.lrate = lrate

    def SharedVariableSetValue(self, SVar, Value):
        for sv, v in zip(SVar,Value):
            sv.grad.data = torch.from_numpy(v).cuda()

    def random_select(self, vec):
        idx_list = [ i for i in range(self.train_data_size) if vec[i] == 0 ]
        M = 0
        numpy.random.shuffle(idx_list)
    
        minibatches_idx = []
        for idx in idx_list:
           if M < self.m0:
              minibatches_idx.append(idx)
              vec[idx] = 1
              M += 1 
           else:
              break

        return minibatches_idx, M

    def gradient_compute(self, batch_idx):
        idx = 0
        cost = 0 
        for i in batch_idx:
            x_data, y_data = self.prepare_data(self.train_data, i)
            x_data = Variable(torch.Tensor(x_data)).cuda()
            y_data = Variable(torch.Tensor(y_data)).cuda()
            y_pred = self.model(x_data)
            c = self.criterion(y_pred, y_data)
            cost += c.data.clone().cpu().numpy()
            c.backward()
            gr = [ p.grad.data.clone().cpu().numpy() for p in self.tparams]
            for p in self.tparams:
                p.grad.data.zero_()
            #print(gr[3])
            g_vector = self.vector_transformation(gr)
            if self.use_diag == True:     
               mat = g_vector**2 
            else:
               mat = numpy.matrix(g_vector).T * numpy.matrix(g_vector)

            if idx ==0:
               grad = gr
               matrix = mat
            else:
               grad = [ numpy.array(gr1) + numpy.array(gr2) for gr1, gr2 in zip(grad, gr)]
               matrix = matrix + mat
        
            idx += 1    
        #hook = []
        #for g in self.tparams:
        #    h = g.register_hook(lambda grad: grad ^ 2)
        #    hook.append(h)
 
        #for i in batch_idx:
        #    x_data, y_data = self.prepare_data(data, i)
        #    y_pred = self.model(x_data)
        #    cost = self.criterion(y_pred, y_data)
        #    cost.backward()
        #    g = [ p.grad for p in self.tparams]
        #if i==0:
        #    comatrix = g
        #else:
        #    comatrix = [gr + g for gr, g in zip(comatrix, g)]

        #for h in hook:
        #    h.remove()

        return cost, grad, matrix
     
    def grad_compute_parallel(self):
        pass

   
       
    def vector_transformation(self, input_vector):
        z = [ numpy.array(g).flatten() for g in input_vector]
        return numpy.array(numpy.concatenate(z))
           
    def train(self,use_optimizer):
        if use_optimizer == 'SGD':
           self.batch_adaptive_sgd_optimizer()
        elif use_optimizer == 'MSGD':
           self.batch_adaptive_momentum_optimizer()
        else:
           print('No such batch adaptive optimization. You can use SGD or MSGD instead.')

    def create_Variables(self):
       self.m = Variable(torch.Tensor([1]), requires_grad = True)
       #self.st = Variable(torch.Tensor([1]), requires_grad = False)
       #self.mu = Variable(torch.Tensor([1]), requires_grad = False)
       #self.lr = Variable(torch.Tensor([1]), requires_grad = False)
       #self.sigma = Variable(torch.Tensor([1]), requires_grad = False)
       #self.value = Variable(torch.Tensor([1]), requires_grad = False)
 
    def compute_mu_and_sigma(self, grad_mean, grad_T ,comatrix, m):
        if self.use_diag == True:
           grad_vector = grad_mean  
           grad_vector_T = grad_mean
        else:
           grad_vector = numpy.matrix(grad_mean)
           grad_vector_T = grad_vector.T
		     
        mu = (grad_mean * grad_T).sum()
        sigma = (grad_vector * comatrix * grad_vector_T).sum()    
        sigma = numpy.sqrt(sigma/m)
        return mu, sigma
    
    def argmax(self, st, mu, sigma, lr):
      start = 1
      end = self.train_data_size 
      mean_f = False
      start_grad = self.grad_compute(start, st, mu, sigma, lr)
      
      if start_grad <= 0:
         return start

      end_grad = self.grad_compute(end, st, mu, sigma, lr)
      #print(start_grad, end_grad)      
      if end_grad >= 0:
         return end

      while start != end:
         mean = (start + end) / 2
         mean_grad = self.grad_compute(mean, st, mu, sigma, lr)
         if mean_grad == 0:
            return mean
         elif mean_grad >0:
            if mean != start:
              start = mean
            else:
              mean_f = True
              break
         else:
            end = mean
     
      if mean_f == True:
         list_s = [mean, mean + 1]
         offset = mean
      elif start == 1:
         list_s = [start, start + 1]
         offset = start
      elif start == self.train_data_size:
         list_s = [start - 1, start]
         offset = start - 1 
      else:
         list_s = [start - 1, start, start + 1]
         offset = start - 1

      g_list = [self.grad_compute(i, st, mu, sigma, lr) for i in list_s]       
      return numpy.argmax(g_list,axis = None) + offset   

    def grad_compute(self, m_value, st_value, mu_value, sigma_value, lrate_value):
        m = Variable(torch.FloatTensor(numpy.array([m_value])), requires_grad = True)
        st = Variable(torch.FloatTensor(numpy.array([st_value])))
        mu = Variable(torch.FloatTensor(numpy.array([mu_value]))) 
        sigma =Variable(torch.FloatTensor(numpy.array([sigma_value])))
        lr = Variable(torch.FloatTensor(numpy.array([lrate_value])))
        a =(st - lr * mu)  / ((((sigma **2) / m) ** 0.5) * lr)
        a_value = a.data.clone()
        a.backward()
        a_grad = m.grad.data.clone()
        m.grad.data.zero_()
        value_of_a = Variable(a_value)
        grad_of_a = Variable(a_grad)

        value = norm.cdf(a_value.numpy())- norm.cdf(-a_value.numpy())
        value = Variable(torch.FloatTensor(numpy.array([value])))
        term1_grad = -(st - (st - lr * mu) * value ) / (m * m)
        term2_grad = -((st - lr*mu) * (2 / math.pi)**0.5 * torch.exp(-value_of_a * value_of_a / 2) * grad_of_a) / m  
        term3 = -lr * ((sigma ** 2 / m) **0.5) * (2 / math.pi) * torch.exp(-value_of_a* value_of_a/2) / m
        term3.backward()
        term3_grad = m.grad.data.clone()
        m.grad.data.zero_()
        total_grad = term1_grad.data + term2_grad.data + term3_grad
        return_grad = total_grad.numpy()
       
        #del total_grad
        return return_grad

      
    def variable_value_set(self, m):
        self.m.data = torch.Tensor([m])
        
    def f_update(self):
        self.optimizer.step()

    def batch_adaptive_sgd_optimizer(self):
        #self.cost_evaluate() 
        seen_samples = 0
        total_samples = 0
        lrate = self.lrate
        while total_samples < self.M:
            vec = numpy.zeros(self.train_data_size, dtype='int8')
            flag = True
            m_alpha = 0
            m_star = -1
            cost = 0
            bound_low  = self.max_batch_size
            while m_alpha < bound_low:
              
               batch_idx, batch_size = self.random_select(vec)
               m_alpha += batch_size
               #compute the sum and square sum of the gradients  
               c, g_mean, g_comatrix = self.gradient_compute(batch_idx)
               #update the accumulate value used later
               if flag == True:
                  g_mean_acc = g_mean 
                  g_comatrix_acc = g_comatrix
                  flag = False
               else:
                  g_mean_acc = [ g_acc + g_m for g_acc, g_m in zip(g_mean_acc,g_mean)]
                  g_comatrix_acc = g_comatrix_acc + g_comatrix
                 
               #compute st
               cost += c
               st = cost / m_alpha             
              
               # compute the total mean gradient
               new_g_mean = [ numpy.array(g_acc) /float(m_alpha) for g_acc  in g_mean_acc]
               #compute the total comatrix
               new_g_mean_vector = self.vector_transformation(new_g_mean)

               if self.use_diag == True:
                   comatrix = new_g_mean_vector **2
               else: 
                   comatrix = numpy.matrix(new_g_mean_vector).T * numpy.matrix(new_g_mean_vector) 
    
               new_g_comatrix = (g_comatrix_acc - m_alpha*  comatrix) / float(m_alpha -1) 
          
             
               #compute mu and sigma
               mu, sigma = self.compute_mu_and_sigma(new_g_mean_vector, new_g_mean_vector, new_g_comatrix, m_alpha)
               if (st - lrate * mu) / (lrate * sigma) > 5:
                  m_star = 1
               else:
                  m_star = self.argmax(st, mu , sigma, lrate)
               bound_low = min(m_star, self.max_batch_size)

            

            self.SharedVariableSetValue(self.tparams, new_g_mean)
            self.f_update()
            #seen_samples += m_alpha
            total_samples += m_alpha
            #print('total_samples:', total_samples)
            #if seen_samples > self.train_data_size:
            #  self.cost_evaluate()
            #  seen_samples = 0 
    def batch_adaptive_momentum_optimizer(self):
        
        momentum = [p.data * 0. for p in self.tparams]
        mu_pt = [p.data * 0. for p in self.tparams]

        lrate = self.lrate
        seen_samples = 0
        gamma = self.gamma
        while seen_samples < self.M: 
            vec = numpy.zeros(self.train_data_size, dtype='int8')
            m_alpha = 0
            m_star = -1
            cost = 0
            bound_low = self.max_batch_size  
            flag = True     
            while m_alpha < bound_low:
                batch_idx, batch_size = self.random_select(vec) 
                m_alpha += batch_size
                #compute the sum and square sum of the gradients  
                c, g_mean, g_comatrix = self.gradient_compute(batch_idx)
               
                #update the accumulate value used later
                if flag == True:
                   g_mean_acc = g_mean 
                   g_comatrix_acc = g_comatrix
                   flag = False
                else:
                   g_mean_acc = [ g_acc + g_m for g_acc, g_m in zip(g_mean_acc,g_mean)]
                   g_comatrix_acc = g_comatrix_acc + g_comatrix
                 
                #compute st
                cost += c
                st = cost / m_alpha             

                new_g_mean = [ g_acc /float(m_alpha) for g_acc  in g_mean_acc]
        
                mu_pt_temp = [gamma * mu_pt_i + ngm for mu_pt_i, ngm in zip(mu_pt, new_g_mean)]
             
                #compute the total comatrix  
                new_g_mean_vector = self.vector_transformation(new_g_mean)
                mu_pt_vector = self.vector_transformation(mu_pt_temp)
			   
                if self.use_diag == True:
                    comatrix = new_g_mean_vector ** 2
                else: 
                    comatrix = numpy.matrix(new_g_mean_vector).T * numpy.matrix(new_g_mean_vector) 
    
                new_g_comatrix = (g_comatrix_acc - m_alpha*  comatrix) / float(m_alpha -1) 
              
                #compute mu and sigma
                mu, sigma = self.compute_mu_and_sigma(new_g_mean_vector, mu_pt_vector, new_g_comatrix, m_alpha)
                    
               
                if (st - lrate * mu) / (lrate * numpy.sqrt(sigma ** 2)) > 5:
                  m_star = 1
                else:
                  m_star = self.argmax(numpy.float32(st), numpy.float32(mu) , numpy.float32(sigma), numpy.float32(lrate))
                bound_low = min(m_star, self.max_batch_size)
 

            mu_pt = [gamma * mu_pt_i + ngm for mu_pt_i, ngm in zip(mu_pt, new_g_mean)]
            momentum = [ lrate * g_m + gamma * mom for g_m, mom in zip(new_g_mean, momentum)]
            
            self.SharedVariableSetValue(self.tparams, momentum)
            self.f_update()
            seen_samples += m_alpha  
    def cost_evaluate(self):
       cost = 0 
       for i in range(self.train_data_size):
           x_data, y_data = self.prepare_data(self.train_data, i)
           x_data = Variable(torch.Tensor(x_data)).cuda()
           y_data = Variable(torch.Tensor(y_data)).cuda()
           y_pred = self.model(x_data)
           loss = self.criterion(y_pred, y_data)
           cost += loss
       cost_file = open('mlp_loss.txt','a')
       cost_file.write(str(cost) + '\n') 
       cost_file.close()
 
