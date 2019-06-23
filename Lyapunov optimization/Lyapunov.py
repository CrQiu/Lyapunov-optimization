
# coding: utf-8

# In[1]:


"""
Lyapunov optimization for P2P net bit rate, 
one-way relay's error rate and two-way relay's outage probability.
Python ver.
"""
import math
import numpy as np
import gym
import time
import EH_P2P

np.random.seed(1)

Looptimes=70000
env=EH_P2P.EH_P2P()
env.Chanpower()
env.Solarread()
B=np.zeros(40)
modulation=0 #0：QPSK, 1:8PSK, 2:16QAM
protocol = 0  #0:DF, 1:AF

#mode: 0->P2P, 1->1-way relay, 2->2-way relay#
mode = 2



for snr in range(0,30,5):
    
    if mode == 0:
        s = env.reset_P2P(snr)
    elif mode == 1:
        s = env.reset_1_way(snr)
    else:
        s = env.reset_2_way(snr)
    V=10000
    Bt=0
    Btt=0
    RR=0
    R=B
    b=0
    RRMAX=0
    RMAX=0
    Sum=0
    aMAX=0
    threshold=300*40**(-snr/10)
    
    print("epoch, Virtual Queue B, battery, average net bit rate, action, snr")
    for t in range (Looptimes): 
        flag = 1
        #print(t)
        judge = 1
        end = 102
        if mode == 1:
            judge = env.judge_1_way()
        elif mode == 2:
            judge = env.judge_2_way(protocol)
            end = 4
            #threshold = 0
        if judge == 1:
            for ii in range (0,end,2):
                a=ii/100
                if mode == 0:
                    r,_ = env.search_P2P([a,modulation])
                    RR=2*V*r-2*Bt*a*b+2*b*a*b-(a*b)**2
                elif mode == 1:
                    r,_ = env.search_1_way(a)
                    RR=-2*V*r-2*Bt*a*b+2*b*a*b-(a*b)**2
                else:
                    a=a*100
                    a = 1
                    r,_ = env.search_2_way([a,protocol])
                    RR=-2*V*r-2*Bt*a*b-(a*b)**2
                    

            
                if flag==1:
                    RRMAX=RR
                    flag=0
                    a_decision=a
                if (RR>RRMAX):
                    a_decision=a
                    RRMAX=RR
        else: a_decision = 0
        #if judge == 1:print(a_decision)
        if mode == 0:
            s_,RMAX,INFO= env.step_P2P([a_decision,modulation])
        elif mode == 1:
            s_,RMAX,INFO= env.step_1_way(a_decision)
        else:
            s_,RMAX,INFO= env.step_2_way([a_decision,protocol])
        #print(RMAX, a_decision)
        if t % 10000==0:
            print(t,Bt,b,Sum/(t+1),a_decision, snr)
        b=s_[2]*300
        Btt=np.maximum(Bt-b+threshold,0)
        Bt=Btt
        Sum+=RMAX
        s=s_
    Results=Sum/Looptimes
    index=(snr+10)/2
    B[int(index)]=Results
    print(Results)


# In[2]:


np.savetxt("Lyap_P2P_1cm"+str(modulation)+".csv", B, delimiter = ',')


# In[3]:


print(B)

