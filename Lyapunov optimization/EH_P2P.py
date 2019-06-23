
# coding: utf-8

# In[8]:

import math
import numpy as np
import csv
import random
from scipy.integrate import quad


def qfunc(x):
        y=0.5*math.erfc(np.sqrt(x/2))
        return y


class EH_P2P(object):
    def __init__(self):
        #######communication settings#######
        self.duration=5*60
        self.Xm=[2,3,4]
        self.alpha=np.array([[1,0],[2/3,2/3],[0.75,0.5]])
        aa=2*((math.sin(math.pi/8))**2)
        bb=2*((math.sin(3*math.pi/8))**2)
        self.beta=np.array([[1,1],[aa,bb],[0.2,1.8]])
        self.Ls=1000
        self.Tp=0.01
        self.Nor_DopplerFreq= 0.05
        self.Max_DopplerFreq= self.Nor_DopplerFreq/self.duration   # unit: Hz
        self.Observe_Data_Iter_Num=(72000)*10
        self.Discount_Factor= 0.99
        
        ############energy settings#########
        self.Solar_cell_size=4*0.2
        self.capacity=12*40*self.duration
        self.state = None
        self.i = 0     #epoch#
                
        ###########one way settings###########
        self.snr2=35
        self.noise2=40/10**(self.snr2/10)
        
          
        
    def Solarread(self):#2010-2011#

        with open("MDP_Optimal_Policy_BER - Multiple Power - Selective DF/ec201006.csv", "r") as csvfile:
            reader = csv.reader(csvfile) # 读取csv文件，返回的是list
            column = [row[5] for row in reader]
            column = np.array(column,dtype=float)

            column = column*0.01
            column = np.reshape(column,(30,-1)).T

            column = column[np.arange(12*7,12*17),:]

        with open("MDP_Optimal_Policy_BER - Multiple Power - Selective DF/ec201106.csv", "r") as csvfile:
            reader = csv.reader(csvfile) # 读取csv文件，返回的是list
            column2=[row[5] for row in reader]
            column2 = np.array(column2,dtype=float)

            column2=column2*0.01
            column2=np.reshape(column2,(30,-1)).T

            column2= column2[np.arange(12*7,12*17),:]


            column3 = np.hstack((column,column2))
            column3 = np.reshape(column3.T,(-1,1))
            column3 = np.tile(column3,(70,1))
            column3 = column3*10*self.duration*self.Solar_cell_size        
            column3= np.maximum(column3,10e-4*np.ones((column3.shape[0],column3.shape[1])))
            
            
            self.solar_sequence = column3
            self.SD_channel_sequence=self.channel_sequence[0:72000]
            self.SR_channel_sequence=self.channel_sequence[72000:72000*2]
            self.RD_channel_sequence=self.channel_sequence[72000*2:72000*3]
            self.channel_1_sequence=self.channel_sequence[0:72000]
            self.channel_2_sequence=self.channel_sequence[72000:72000*2]


        return column3
        csvfile.close()
        
    def Solartest(self):
        with open("MDP_Optimal_Policy_BER - Multiple Power - Selective DF/ec201206.csv", "r") as csvfile:
            reader = csv.reader(csvfile) # 读取csv文件，返回的是list
            column = [row[5] for row in reader]
            column = np.array(column,dtype=float)

            column = column*0.01
            column = np.reshape(column,(30,-1)).T

            column = column[np.arange(12*7,12*17),:]#7am to 5pm#
            column = np.reshape(column,(-1,1))
            column3 = np.tile(column,(41,1))               
            
            self.solar_sequence = column3*10*self.duration*self.Solar_cell_size
            column3= np.maximum(column3,10e-4*np.ones((column3.shape[0],column3.shape[1])))


            self.SD_channel_sequence=self.channel_sequence[0:72000]
            self.SR_channel_sequence=self.channel_sequence[72000:72000*2]
            self.RD_channel_sequence=self.channel_sequence[72000*2:72000*3]
            self.channel_1_sequence=self.channel_sequence[0:72000]
            self.channel_2_sequence=self.channel_sequence[72000:72000*2]

     

        return column3
        csvfile.close()


    def solarnoise(self):
        self.solar_sequence += 1*np.random.randn(self.solar_sequence.shape[0],self.solar_sequence.shape[1])          
        self.solar_sequence= np.maximum(self.solar_sequence,np.zeros((self.solar_sequence.shape[0],self.solar_sequence.shape[1])))
            



    def Chanpower(self):#Jakes model#
        n0= 100
        np2= (2*n0+1)*2
        wm= 2*math.pi*self.Max_DopplerFreq
        rp= 2.0*math.pi*np.random.rand(1,n0)
        Bn= math.pi*np.arange(1,n0+1)/n0
        Wn= wm*np.cos(2*math.pi*np.arange(1,n0+1)/np2)
        tt= np.arange(0,(self.Observe_Data_Iter_Num))*self.duration
        xc1_temp= np.kron(np.reshape(np.ones(np.size(tt)),(1,-1)),np.cos(np.reshape(Bn,(1,-1)).T))*np.cos(np.reshape(Wn,(1,-1)).T.dot(np.reshape(tt,(1,-1)))+np.kron(np.ones(np.size(tt)),rp.T)) 
        xs1_temp= np.kron(np.reshape(np.ones(np.size(tt)),(1,-1)),np.sin(np.reshape(Bn,(1,-1)).T))*np.cos(np.reshape(Wn,(1,-1)).T.dot(np.reshape(tt,(1,-1)))+np.kron(np.ones(np.size(tt)),rp.T)) 
        xc1= sum(xc1_temp)
        xs1= sum(xs1_temp)
        xc= 2.0/np.sqrt(np2)*np.sqrt(2.0)*xc1 + 2.0/np.sqrt(np2)*np.cos(math.pi/4)*np.cos(wm*tt)
        xs= 2.0/np.sqrt(np2)*np.sqrt(2.0)*xs1 + 2.0/np.sqrt(np2)*np.sin(math.pi/4)*np.cos(wm*tt)
        Observe_Channel_Sequence= xc**2+xs**2 # instantaneous channel power 
        #print(numpy.mean(Observe_Channel_Sequence),xc1.shape)
        self.channel_sequence = Observe_Channel_Sequence
        return Observe_Channel_Sequence
    
    

    
    def search_P2P(self,act): #for Lyapunov optimization#
        state=self.state
        solar,channel,battery=state       
        battery2=battery*self.duration
        channel2=(1+channel)
        action,Modulation_type=act
        error_rate=self.alpha[Modulation_type,0]*qfunc(self.beta[Modulation_type,0]*action*battery2*channel2/self.noise/self.duration)+self.alpha[Modulation_type,1]*qfunc(self.beta[Modulation_type,1]*action*battery2*channel2/self.noise/self.duration)
        reward=(self.Xm[Modulation_type]*self.Ls/self.Tp)*(1-error_rate)**(self.Xm[Modulation_type]*self.Ls)
        return  reward, {}


      
    

    def step_P2P(self,act):
        ##recover from normalization##
        state=self.state
        solar,channel,battery=state       
        solar2=solar      
        solar2*=self.duration
        battery2=battery*self.duration
        channel2=channel
        ##recover from normalization##
        action,Modulation_type=act
        SNR=action*battery2*channel2/self.noise/self.duration

        error_rate=self.alpha[Modulation_type,0]*qfunc(self.beta[Modulation_type,0]*SNR)+self.alpha[Modulation_type,1]*qfunc(math.sqrt(self.beta[Modulation_type,1]*SNR))
            
        reward=(self.Xm[Modulation_type]*self.Ls/self.Tp)*(1-error_rate)**(self.Xm[Modulation_type]*self.Ls)#/100-200

        self.i+=1
        #normalize#
        battery=np.minimum(battery2*(1-action)+solar2,self.capacity)/self.duration
        channel=(self.channel_sequence[self.i])
        solar=self.solar_sequence[self.i]/self.duration
        self.state=solar,(channel),battery
        #normalize#
        return np.array(self.state), reward, {}    

    





    def judge_1_way(self):#whether the relay is on or off#
        state=self.state
        solar,SD_channel,SR_channel,RD_channel,battery=state


        solar2=solar*self.duration

        battery2=battery*self.duration

        SD_channel2=SD_channel
        SR_channel2=SR_channel
        RD_channel2=RD_channel
        Num_ModulationType= 4; # M-PSK: 4(Q)-PSK 
        G_MPSK_Modulation= math.sin(math.pi/Num_ModulationType)**2
        Decoding_Capability_dB= 15; # 30;15;  (dB)
        Decoding_Capability= 10**(Decoding_Capability_dB/10)
        
        Source_Transmit_Power= 40
        if (Source_Transmit_Power<Decoding_Capability/SR_channel*self.noise2):
            return 0 
        else:
            return 1


    def search_1_way(self,action):
        state=self.state
        solar,SD_channel,SR_channel,RD_channel,battery=state
        flag=0

        solar2=solar
        solar2*=self.duration
        battery2=battery*self.duration
        SD_channel2=SD_channel
        SR_channel2=SR_channel
        RD_channel2=RD_channel


        Num_ModulationType= 4; # M-PSK: 4(Q)-PSK 
        G_MPSK_Modulation= math.sin(math.pi/Num_ModulationType)**2
        Decoding_Capability_dB= 15; # 30;15;  (dB)
        Decoding_Capability= 10**(Decoding_Capability_dB/10)
        
        Source_Transmit_Power= 40

        if (Source_Transmit_Power<Decoding_Capability/SR_channel*self.noise2):
            reward=quad(lambda x:np.exp(-G_MPSK_Modulation*Source_Transmit_Power*SD_channel2/self.noise/(np.sin(x))**2)/math.pi,0,math.pi*(Num_ModulationType-1)/Num_ModulationType)[0]
     
        else:
            reward=quad(lambda x:np.exp(-G_MPSK_Modulation*(Source_Transmit_Power*SD_channel2+RD_channel2*action*battery2/self.duration*2)/self.noise/(np.sin(x))**2)/math.pi,0,math.pi*(Num_ModulationType-1)/Num_ModulationType)[0]

        return reward, None
    
    
    def step_1_way(self,action):
        state=self.state
        solar,SD_channel,SR_channel,RD_channel,battery=state
        flag=0

        solar2=solar
        solar2*=self.duration
        battery2=battery*self.duration
        SD_channel2=SD_channel
        SR_channel2=SR_channel
        RD_channel2=RD_channel


        Num_ModulationType= 4; # M-PSK: 4(Q)-PSK 
        G_MPSK_Modulation= math.sin(math.pi/Num_ModulationType)**2
        Decoding_Capability_dB= 15; # 30;15;  (dB)
        Decoding_Capability= 10**(Decoding_Capability_dB/10)
        
        Source_Transmit_Power= 40

        if (Source_Transmit_Power<Decoding_Capability/SR_channel*self.noise2):
            reward=quad(lambda x:np.exp(-G_MPSK_Modulation*Source_Transmit_Power*SD_channel2/self.noise/(np.sin(x))**2)/math.pi,0,math.pi*(Num_ModulationType-1)/Num_ModulationType)[0]
     
        else:
            reward=quad(lambda x:np.exp(-G_MPSK_Modulation*(Source_Transmit_Power*SD_channel2+RD_channel2*action*battery2/self.duration*2)/self.noise/(np.sin(x))**2)/math.pi,0,math.pi*(Num_ModulationType-1)/Num_ModulationType)[0]


        self.i+=1
        battery=np.minimum(battery2*(1-action)+solar2,self.capacity)/self.duration
        SD_channel=self.SD_channel_sequence[self.i]
        SR_channel=self.SR_channel_sequence[self.i]
        RD_channel=self.RD_channel_sequence[self.i]
        solar=self.solar_sequence[self.i]/self.duration
        self.state=solar,SD_channel,SR_channel,RD_channel,battery
        return np.array(self.state), reward, flag



    def judge_2_way(self,protocol):
        state=self.state
        solar,channel_1,channel_2,battery=state
        solar2=(solar)*self.duration
        battery2=battery*self.duration
        channel_12=channel_1
        channel_22=channel_2
        Source_Transmit_Power= self.Pab*35
        
        R1=2
        R2=2
        if protocol==0:
            cond1=channel_12*Source_Transmit_Power/self.noise-2**(2*R1)+1
            cond2=channel_22*Source_Transmit_Power/self.noise-2**(2*R2)+1
            cond3=(channel_12+channel_22)*Source_Transmit_Power/self.noise-2**(2*R1+2*R2)+1
            if cond1 <= 0 :
               #print("1",self.noise2,cond1)
               return 0
            elif cond2 <= 0 :
               #print("2")
               return 0
            elif cond3 <= 0 :
               #print("3")
               return 0
            elif battery2<np.maximum((2**(2*R1)-1)*self.noise/channel_22,(2**(2*R2)-1)*self.noise/channel_12)/2:
               #print("4")
               return 0

            else: return 1
        else:
            cond1=channel_12*Source_Transmit_Power/self.noise-2**(2*R1)+1
            cond2=channel_22*Source_Transmit_Power/self.noise-2**(2*R2)+1
            threshold1=(2**(2*R1)-1)*self.noise*(channel_12*Source_Transmit_Power+channel_22*Source_Transmit_Power+self.noise)/(channel_12*channel_22*Source_Transmit_Power-(2**(2*R1)-1)*self.noise*channel_22)
            threshold2=(2**(2*R2)-1)*self.noise*(channel_12*Source_Transmit_Power+channel_22*Source_Transmit_Power+self.noise)/(channel_12*channel_22*Source_Transmit_Power-(2**(2*R2)-1)*self.noise*channel_12)
            if cond1<=0 or cond2<=0:
                return 0
            elif battery2<np.maximum(threshold1,threshold2)/2:
                return 0
            else: return 1



    def search_2_way(self,act):
        state=self.state
        solar,channel_1,channel_2,battery=state
        solar2=(solar)*self.duration
        battery2=battery*self.duration
        channel_12=channel_1
        channel_22=channel_2
        Source_Transmit_Power= self.Pab*35
        action,protocol=act
        R1=2
        R2=2
        if protocol==0:
                #print("cond1")
            if action>=1:
                real_act=np.maximum((2**(2*R1)-1)*self.noise/channel_22,(2**(2*R2)-1)*self.noise/channel_12)*self.duration/2
                reward = 0
                #print("cond2")
            else: 
                real_act=0
                reward = 1
        else:
            threshold1=(2**(2*R1)-1)*self.noise*(channel_12*Source_Transmit_Power+channel_22*Source_Transmit_Power+self.noise)/(channel_12*channel_22*Source_Transmit_Power-(2**(2*R1)-1)*self.noise*channel_22)
            threshold2=(2**(2*R2)-1)*self.noise*(channel_12*Source_Transmit_Power+channel_22*Source_Transmit_Power+self.noise)/(channel_12*channel_22*Source_Transmit_Power-(2**(2*R2)-1)*self.noise*channel_12)            
                #print("cond1")    
            if action>=1:
                real_act= np.maximum(threshold1,threshold2)*self.duration/2
                reward = 0
                #print("cond2")
            else: 
                real_act=0
                reward = 1
        return reward,{}

    def step_2_way(self,act):
        state=self.state
        solar,channel_1,channel_2,battery=state
        solar2=(solar)*self.duration
        battery2=battery*self.duration
        channel_12=channel_1
        channel_22=channel_2
        Source_Transmit_Power= self.Pab*35
        action,protocol=act
        R1=2
        R2=2
        if protocol==0:
                #print("cond1")
            if action>=1:
                real_act=np.maximum((2**(2*R1)-1)*self.noise/channel_22,(2**(2*R2)-1)*self.noise/channel_12)*self.duration/2
                reward = 0
                #print("cond2")
            else: 
                real_act=0
                reward = 1
        else:
            threshold1=(2**(2*R1)-1)*self.noise*(channel_12*Source_Transmit_Power+channel_22*Source_Transmit_Power+self.noise)/(channel_12*channel_22*Source_Transmit_Power-(2**(2*R1)-1)*self.noise*channel_22)
            threshold2=(2**(2*R2)-1)*self.noise*(channel_12*Source_Transmit_Power+channel_22*Source_Transmit_Power+self.noise)/(channel_12*channel_22*Source_Transmit_Power-(2**(2*R2)-1)*self.noise*channel_12)            
                #print("cond1")    
            if action>=1:
                real_act= np.maximum(threshold1,threshold2)*self.duration/2
                reward = 0
                #print("cond2")
            else: 
                real_act=0
                reward = 1
        self.i+=1
        battery=np.minimum(battery2-real_act+solar2,126000)/self.duration
        channel_1=self.SD_channel_sequence[self.i]
        channel_2=self.SR_channel_sequence[self.i]
        solar=self.solar_sequence[self.i]/self.duration
        self.state=solar,channel_1,channel_2,battery
        return np.array(self.state),reward,{}



    def reset_P2P(self,snr):
        self.i=0
        self.state = np.zeros((3,))
        self.snr=snr
        self.noise=10**(-snr/10)
        self.state=self.solar_sequence[self.i+1]/self.duration,(self.channel_sequence[self.i]),self.solar_sequence[self.i]/self.duration
        return np.array(self.state)
    

    def reset_1_way(self,snr):
        self.state = np.zeros((5,))
        self.i=0
        self.state=self.solar_sequence[self.i+1]/self.duration,self.SD_channel_sequence[self.i],self.SR_channel_sequence[self.i],self.RD_channel_sequence[self.i],self.solar_sequence[self.i]/self.duration
        self.noise=40/10**(snr/10)
        return np.array(self.state)


    def reset_2_way(self,snr):
        self.i=0
        self.state = np.zeros((4,))
        self.Chanpower()
        self.Solarread()
        self.state=self.solar_sequence[self.i+1]/self.duration,self.channel_1_sequence[self.i],self.channel_2_sequence[self.i],self.solar_sequence[self.i]/self.duration
        CN_Ratio_dB= snr+ 10*math.log10(35)
        ChantoNoise_Ratio= 10**(CN_Ratio_dB/10)
        self.noise= 35/ChantoNoise_Ratio
        #print(self.noise)
        self.Pab = 3; # power of A and B
        #  We are very sorry because there is something wrong with the simulation settings. The correct source power is Ps = 3*35mW,  #
        #  and so is it in the benchmark paper 'On Outage Probability for Two-Way Relay Networks With Stochastic Energy Harvesting'.#
        return np.array(self.state)