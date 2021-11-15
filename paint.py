import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
plt.switch_backend('agg')
plt.style.use('ggplot')
f1=open('D:\\dqn\\dqn-breakout\\rewards.txt','r')
f2=open('D:\\dqn\\dqn-breakout\\state_machine.txt','r')
f3=open('D:\\dqn\\dqn-breakout\\Duel.txt','r')
f4=open('D:\\dqn\\dqn-breakout\\Duel+DDQN.txt','r')
f5=open('D:\\dqn\\dqn-breakout\\Duel+DDQN+epsilon=0.txt','r')
f6=open('D:\\dqn\\dqn-breakout\\duel+ddqn+prioritised reward.txt','r')
f7=open('D:\\dqn\\dqn-breakout\\modified convolution layer.txt','r')
f8=open('D:\\dqn\\dqn-breakout\\only ddqn.txt','r')

def avg(r):
    return [np.mean(r[i-100:i]) for i in range(100, len(r))]

x1=[]
x2=[]
x3=[]
x4=[]
x5=[]
x6=[]
x7=[]
x8=[]
y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
y6=[]
y7=[]
y8=[]
cnt6=0
cnt7=0

for line in f1:
    s=line.split()
    l=list(float(i) for i in s)
    x1.append(l[0])
    y1.append(l[2])
for line in f2:
    s=line.split()
    l=list(float(i) for i in s)
    x2.append(l[0])
    y2.append(l[2])
for line in f3:
    s=line.split()
    l=list(float(i) for i in s)
    x3.append(l[0])
    y3.append(l[2])
for line in f4:
    s=line.split()
    l=list(float(i) for i in s)
    x4.append(l[0])
    y4.append(l[2])
for line in f5:
    s=line.split()
    l=list(float(i) for i in s)
    x5.append(l[0])
    y5.append(l[2])
for line in f6:
    s=line.split()
    l=list(float(i) for i in s)
    x6.append(cnt6)
    y6.append(l[2])
    cnt6+=1
for line in f7:
    s=line.split()
    l=list(float(i) for i in s)
    x7.append(cnt7)
    y7.append(l[2])
    cnt7+=1
for line in f8:
    s=line.split()
    l=list(float(i) for i in s)
    x8.append(l[0])
    y8.append(l[2])
    
avg_y1=avg(y1)
avg_y2=avg(y2)
avg_y3=avg(y3)
avg_y4=avg(y4)
avg_y5=avg(y5)
avg_y6=avg(y6)
avg_y7=avg(y7)
avg_y8=avg(y8)

idx1=list(range(100,len(y1)))
idx2=list(range(100,len(y2)))
idx3=list(range(100,len(y3)))
idx4=list(range(100,len(y4)))
idx5=list(range(100,len(y5)))
idx6=list(range(100,len(y6)))
idx7=list(range(100,len(y7)))
idx8=list(range(100,len(y8)))

baseline_color=(120/255,10/255,153/255,0.5)
plt.figure(num=1,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x1,y1,color=(1,99/255,20/255,0.35),label='Rewards of basic implementation')
plt.plot(idx1,avg_y1,color=(1,99/255,20/255,1),label='Avg of rewards over 100 batches')
#plt.plot(idx1,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_basic.png')

plt.figure(num=2,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x2,y2,color=(99/255,1,20/255,0.35),label='Rewards of state machine')
plt.plot(idx2,avg_y2,color=(99/255,1,20/255,1),label='Avg of rewards over 100 batches')
#plt.plot(idx2,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_state_machine.png')

plt.figure(num=3,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x3,y3,color=(20/255,1,99/255,0.35),label='Rewards of Duel')
plt.plot(idx3,avg_y3,color=(20/255,1,99/255,1),label='Avg of rewards over 100 batches')
#plt.plot(idx3,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_duel.png')

plt.figure(num=4,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x4,y4,color=(20/255,99/255,1,0.35),label='Rewards of Duel+DDQN')
plt.plot(idx4,avg_y4,color=(20/255,99/255,1,1),label='Avg of rewards over 100 batches')
#plt.plot(idx3,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_Duel+DDQN.png')

plt.figure(num=5,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x5,y5,color=(99/255,20/255,1,0.35),label='Rewards of Duel+DDQN+eps=0')
plt.plot(idx5,avg_y5,color=(99/255,20/255,1,1),label='Avg of rewards over 100 batches')
#plt.plot(idx3,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_Duel+DDQN+eps=0.png')

plt.figure(num=7,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x6,y6,color=(122/255,165/255,215/255,0.35),label='Rewards of duel+ddqn+prioritised')
plt.plot(idx6,avg_y6,color=(122/255,165/255,215/255,1),label='Avg of rewards over 100 batches')
#plt.plot(idx3,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_duel+ddqn+prioritised.png')

plt.figure(num=8,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x7,y7,color=(1,1,0,0.35),label='Rewards of modified convolution layer')
plt.plot(idx7,avg_y7,color=(1,1,0,1),label='Avg of rewards over 100 batches')
#plt.plot(idx3,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_modified convolution layer.png')

plt.figure(num=9,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(x8,y8,color=(116/255,0,0,0.35),label='Rewards of ddqn')
plt.plot(idx8,avg_y8,color=(116/255,0,0,1),label='Avg of rewards over 100 batches')
#plt.plot(idx3,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_ddqn.png')

plt.figure(num=6,figsize=(10,6),dpi=150)
plt.xlabel("Batches with the size of 10000 steps")
plt.ylabel("Avg of rewards")
plt.title('Rewards Over Steps')
plt.plot(idx1,avg_y1,color=(1,99/255,20/255,1),label='Avg of rewards of basic implementation')
plt.plot(idx2,avg_y2,color=(99/255,1,20/255,1),label='Avg of rewards of state machine')
plt.plot(idx3,avg_y3,color=(20/255,1,99/255,1),label='Avg of rewards of Duel')
plt.plot(idx4,avg_y4,color=(20/255,99/255,1,1),label='Avg of rewards of Duel+DDQN')
plt.plot(idx5,avg_y5,color=(99/255,20/255,1,1),label='Avg of rewards of Duel+DDQN+eps=0')
plt.plot(idx6,avg_y6,color=(122/255,165/255,215/255,1),label='Avg of rewards of duel+ddqn+prioritised')
plt.plot(idx7,avg_y7,color=(1,1,0,1),label='Avg of rewards of modified convolution layer')
plt.plot(idx8,avg_y8,color=(116/255,0,0,1),label='Avg of rewards of ddqn')
#plt.plot(idx3,[30 for _ in idx1],color=(20/255,83/255,1,0.5),label='Baseline(avg of rewards = 30)')
plt.legend()
plt.savefig('rewards_total.png')

