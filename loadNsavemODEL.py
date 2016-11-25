# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:01:01 2016

@author: ceciliaLee
"""

""" Loading and saving model
     1. SAVE those weights
     2. SAVE current-best estimates as the search progresses
     
    
     Do not pickle your training or test functions for long-term storage
     
 """
 
 ##  Pickle the numpy ndarrays from shared variables
import cPickle
save_file = open('path', 'wb')   #  this will overwrite current contents
cPickle.dump(w.get_value(borrow=True), save_file, -1) # the -1 is for HIGHEST_PROTOCOL
cPickle.dump(v.get_value(borrow=True), save_file, -1) # .. and it triggers much more
cPickle.dump(u.get_value(borrow=True), save_file, -1) # .. storage than numpy’s default
save_file.close()
 
 ## load data

load_file = open('path')
 
w.set_value(cPickle.load(save_file), borrow=True)
v.set_value(cPickle.load(save_file), borrow=True)
u.set_value(cPickle.load(save_file), borrow=True)