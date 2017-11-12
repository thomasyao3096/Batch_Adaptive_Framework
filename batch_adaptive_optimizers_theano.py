
from __future__ import print_function

from collections import OrderedDict
import math
import numpy 
import theano
from theano import config
import theano.tensor as tensor
from scipy.stats import norm

class batch_adaptive_optimizer():
    def __init__(self,  tparams, train_data, train_data_size, valid_data, valid_data_size, prepare_data, f_cost, f_grad, 
    	f_pred, use_data, M, max_batch_size, lrate=0.05, gamma=0.9, m0=30, use_diag=True):
       # learning rate for sgd, momemtum ..
       self.lrate = lrate

       # an ordered dictionary, each value is a shared numpy array
       self.tparams = tparams  

       # training data set, parameter for function prepare_data, 
       # should be a tuple containing two list (list of X and list of label Y)
       self.train_data = train_data  

       # the size of training data set
       self.train_data_size = train_data_size  

       # validation set, parameter for function prepare_data
       self.valid_data = valid_data
  
       # the size of validation set
       self.valid_data_size = valid_data_size 

       # the minimum increment for the increasing batch size
       self.m0 = m0
  
       # a function that takes data set (train_data or valid_data) and the index of 
       # a certain piece of data (ranging from 0 to the size of the data set) as input, and produce (X, Y) as output, 
       # label Y is a one-hot vector
       self.prepare_data = prepare_data
  
       # theano function, takes X and Y as input and produce cost
       self.f_cost = f_cost
  
       # theano function, takes X and Y as input and produce the gradients
       self.f_grad = f_grad
  
       # the name of the data set, for record purpose
       self.use_data= use_data

       # the total budget
       self.M = M

       # the max size of each batch
       self.max_batch_size = max_batch_size 

       # whether to use diagonal to replace the covariance matrix, default True
       self.use_diag = use_diag 
 
       # the hyper parameter for momentum
       self.gamma = gamma  
            
    def train(self,use_optimizer):
        if use_optimizer == 'MSGD':
            self.batch_adaptive_momentum_sgd_optimizer()
        elif use_optimizer == 'SGD':
            self.batch_adaptive_sgd_optimizer()       

    def SharedVariableSetValue(self, SVar, Value):
        for sv, v in zip(SVar,Value):
            sv.set_value(v)
         
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

    def vector_transformation(self, input_vector):
        z = [ numpy.array(g).flatten() for g in input_vector]
        return numpy.array(numpy.concatenate(z))
		
    def gradient_compute(self, batch_idx):
        for i in range(len(batch_idx)):
           input_tuple = self.prepare_data(self.train_data, batch_idx[i])
           gr = self.f_grad(*input_tuple)
           g_vector = self.vector_transformation(gr)
           if self.use_diag == True:     
             mat = g_vector**2 
           else:
             mat = numpy.matrix(g_vector).T * numpy.matrix(g_vector)
           if i ==0:
             grad = gr
             matrix = mat
           else:
             grad = [ numpy.array(gr1) + numpy.array(gr2) for gr1, gr2 in zip(grad, gr)]
             matrix = matrix + mat
        return grad, matrix

    def cost_compute(self, batch_idx):
        cost = 0
        for i in batch_idx:
            input_tuple = self.prepare_data(self.train_data, i)
  
            c = self.f_cost(*input_tuple)
            cost = cost + c
     
        return cost

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

    def batch_adaptive_momentum_sgd_optimizer(self):
        a_f, argmax_f = self.create_argmax_function()
        f_update, gsum= self.create_momentum_upgrade_function()

        momentum = [p.get_value() * 0. for k , p in self.tparams.items()]
        mu_pt = [p.get_value() * 0. for k, p in self.tparams.items()]

        update_count = 0
        total_update_count = 0
        seen_samples = 0
        epoch = 0

        lrate = self.lrate

        gamma = self.gamma
        
        while seen_samples < self.M:
            vec = numpy.zeros(self.train_data_size, dtype='int8')
            m_star = -1
            cost = 0
            m_alpha = 0
            bound_low = self.max_batch_size
            flag = True

            while m_alpha < bound_low:
               #random select some samples from the rest training set
               batch_idx, batch_size = self.random_select(vec)
               if batch_size == 0:
                  break

               m_alpha += batch_size

               #compute the sum and square sum of the gradients  
               g_mean, g_comatrix = self.gradient_compute(batch_idx)
               
               if flag == True:
                   flag = False
                   g_mean_acc = g_mean 
                   g_comatrix_acc = g_comatrix
               else:
                   g_mean_acc = [ g_acc + g_m for g_acc, g_m in zip(g_mean_acc,g_mean)]
                   g_comatrix_acc = g_comatrix_acc + g_comatrix
                 
             
               #compute st
               c = self.cost_compute(batch_idx) 
               cost += c
               st = cost / m_alpha
             
               # compute the total mean gradient
               new_g_mean = [ numpy.array(g_acc)/float(m_alpha) for g_acc  in g_mean_acc]
        
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
                 m_star = self.argmax(numpy.float32(st), numpy.float32(mu) , numpy.float32(sigma), numpy.float32(0), numpy.float32(lrate), argmax_f, a_f)
               bound_low = min(m_star, self.max_batch_size)
 

            mu_pt = [gamma * mu_pt_i + ngm for mu_pt_i, ngm in zip(mu_pt, new_g_mean)]
            momentum = [ lrate * g_m + gamma * mom for g_m, mom in zip(new_g_mean, momentum)]
            
            self.SharedVariableSetValue(gsum, momentum)
            f_update()
            seen_samples += m_alpha 
           
    def create_argmax_function(self):
       m = tensor.scalar('m', dtype = 'int64')
       st = tensor.scalar('st', dtype = config.floatX)
       mu = tensor.scalar('mu', dtype = config.floatX)
       lr = tensor.scalar('lr', dtype = config.floatX)
       sigma_t = tensor.scalar('sigma', dtype = config.floatX)
       sigma_t_prev = tensor.scalar('sigma_prev', dtype = config.floatX)
       value = tensor.scalar('value', dtype = config.floatX)

       a = (st - lr * mu)  / (numpy.sqrt(sigma_t ** 2 / m + sigma_t_prev ** 2) * lr)
       a_f = theano.function([m, st, mu, sigma_t, sigma_t_prev, lr], a)

       a_grad = tensor.grad(a, m)
    
       term1_grad = -(st - (st - lr * mu) * value ) / (m * m)

       term2_grad = - (st- lr * mu) * numpy.sqrt(2 / math.pi) * numpy.exp(-a*a/2) * a_grad / m 

       term3 = - lr * numpy.sqrt(sigma_t ** 2 / m + sigma_t_prev ** 2) * numpy.sqrt(2 / math.pi) * numpy.exp(-a*a/2) / m
       term3_grad = tensor.grad(term3, m)
        
       E = term1_grad + term2_grad + term3_grad
       argmax_f = theano.function([m, st, mu, sigma_t, sigma_t_prev, lr, value], E)

       return a_f, argmax_f


    def argmax(self, st, mu, sigma, sigma_prev, lr, argmax_f, a_f):
      start = 1
      end = self.train_data_size 
      mean_f = False
      start_grad = self.grad_compute(start, st, mu, sigma, sigma_prev, lr, argmax_f, a_f)
      
      if start_grad <= 0:
         return start

      end_grad = self.grad_compute(end, st, mu, sigma, sigma_prev, lr, argmax_f, a_f)
      
      if end_grad >= 0:
         return end

      while start != end:
         mean = (start + end) / 2
         mean_grad = self.grad_compute(mean, st, mu, sigma, sigma_prev, lr, argmax_f, a_f)
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

      g_list = [self.grad_compute(i, st, mu, sigma, sigma_prev, lr, argmax_f, a_f) for i in list_s]       
      return numpy.argmax(g_list,axis = None) + offset     

    def create_momentum_upgrade_function(self):
        gsum = [theano.shared(p.get_value() * 0., name='%s_grad_sum' % k)
               for k, p in self.tparams.items()]

        pup = [(p, p - mom) for p, mom in zip(self.tparams.values(), gsum)]

        # Function that updates the weights from the previously computed
        # gradient.
        f_update = theano.function([], [], updates=pup,
                               name='basgd_f_update')

        return f_update, gsum
    
    def create_sgd_upgrade_function(self):
       gsum = [theano.shared(p.get_value() * 0., name='%s_grad_sum' % k)
               for k, p in self.tparams.items()]
       
       lrate = [theano.shared(p.get_value() * 0. + self.lrate, name='%s_lrate' % k)
               for k, p in self.tparams.items()]
       
       lr = tensor.scalar("lr", dtype =  config.floatX)
       pup = [(p, p - lr * g) for p, g in zip(self.tparams.values(), gsum)]

       f_update = theano.function([lr], [], updates=pup,
                               name='basgd_f_update')

       return f_update, gsum

    def grad_compute(self, m, st, mu, sigma, sigma_prev, lr, argmax_f, a_f):
        a = a_f(m, st, mu, sigma, sigma_prev, lr)
        value = numpy.float32(norm.cdf(a)- norm.cdf(-a))
        return argmax_f(m, st, mu, sigma, sigma_prev, lr, value)


    def batch_adaptive_sgd_optimizer(self):
        a_f, argmax_f = self.create_argmax_function()
        f_update, gsum = self.create_sgd_upgrade_function()
        update_count = 0
        total_update_count = 0
        seen_samples = 0
        lrate = self.lrate
        epoch = 0
        
        while seen_samples < self.M:
          vec = numpy.zeros(self.train_data_size, dtype='int8')
          m_star = -1
          cost = 0
          m_alpha = 0
          bound_low = self.max_batch_size
          flag = True    
          
          while m_alpha < bound_low:
             #random select some samples from the rest training set
             batch_idx, batch_size = self.random_select(vec)
             if batch_size == 0:
                break

             m_alpha += batch_size
             #compute the sum and square sum of the gradients  
             g_mean, g_comatrix = self.gradient_compute(batch_idx)
               
             #update the accumulate value used later
             if flag == True:
                 g_mean_acc = g_mean 
                 g_comatrix_acc = g_comatrix
                 flag = False
             else:
                g_mean_acc = [ g_acc + g_m for g_acc, g_m in zip(g_mean_acc,g_mean)]
                g_comatrix_acc = g_comatrix_acc + g_comatrix
                 
             #compute st
             c = self.cost_compute(batch_idx) 
             cost += c
             st = cost / m_alpha
             
             # compute the total mean gradient
             new_g_mean = [ numpy.array(g_acc)/float(m_alpha) for g_acc  in g_mean_acc]
             
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
               m_star = self.argmax(numpy.float32(st), numpy.float32(mu) , numpy.float32(sigma), numpy.float32(0), numpy.float32(lrate), argmax_f, a_f)
             bound_low = min(m_star, self.max_batch_size)

          self.SharedVariableSetValue(gsum, new_g_mean)
          f_update(lrate)
          seen_samples += m_alpha
