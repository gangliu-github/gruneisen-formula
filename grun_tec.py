#!/usr/bin/python
import numpy as np 
from numpy import linalg as LA
pa= open('grun_tec.in', 'r')
An=(pa.readline()).strip()
Dm=(pa.readline()).strip()
IL=(pa.readline()).rstrip()
B=(pa.readline()).rstrip()
dd=(pa.readline()).rstrip()
bi=(pa.readline()).rstrip()
fn0=(pa.readline()).rstrip()
fn1=(pa.readline()).rstrip()
fn2=(pa.readline()).rstrip()
fn3=(pa.readline()).rstrip()
fn4=(pa.readline()).rstrip()
fn5=(pa.readline()).rstrip()
fn6=(pa.readline()).rstrip()
c11=(pa.readline()).rstrip()
c22=(pa.readline()).rstrip()
c33=(pa.readline()).rstrip()
c12=(pa.readline()).rstrip()
c13=(pa.readline()).rstrip()
c23=(pa.readline()).rstrip()
Tm=(pa.readline()).rstrip()
v0=(pa.readline()).rstrip()
pa.close()
#-----------------------------------------
pi=np.pi
kb=1.381e-23
hb=1.055e-34
An= An.split('=')[1].split(',')[0].strip()
Dm= float(Dm.split('=')[1].split(',')[0].strip())
IL= float(IL.split('=')[1].split(',')[0].strip())
B= float(B.split('=')[1].split(',')[0].strip())
dd= float(dd.split('=')[1].split(',')[0].strip())
bi= float(bi.split('=')[1].split(',')[0].strip())
fn0= fn0.split('=')[1].split(',')[0].strip()
fn1= fn1.split('=')[1].split(',')[0].strip()
fn2= fn2.split('=')[1].split(',')[0].strip()
fn3= fn3.split('=')[1].split(',')[0].strip()
fn4= fn4.split('=')[1].split(',')[0].strip()
fn5= fn5.split('=')[1].split(',')[0].strip()
fn6= fn6.split('=')[1].split(',')[0].strip()
c11= float(c11.split('=')[1].split(',')[0].strip())
c22= float(c22.split('=')[1].split(',')[0].strip())
c33= float(c33.split('=')[1].split(',')[0].strip())
c12= float(c12.split('=')[1].split(',')[0].strip())
c13= float(c13.split('=')[1].split(',')[0].strip())
c23= float(c23.split('=')[1].split(',')[0].strip())
Tm= int(Tm.split('=')[1].split(',')[0].strip())
v0= float(v0.split('=')[1].split(',')[0].strip())*1e-30
#--------------------------------------------------
def macrogrun(f0,f1,f2,w,dd,T,jm):
 mode_g=-1/f0*(f2-f1)/(2*dd)
 c1=(hb*f0/kb/T)
 c2=np.exp(-c1)
 cv=w*kb*c1**2*c2/(1+c2**2-2*c2)
 Cv=cv.sum()/w.sum()*jm
 I=cv*mode_g
 MG=I.sum()/cv.sum()
 return MG,Cv
#-------------------------------------------------------
def extract(fn,jm):
 j=1
 ff=0
 tt=np.zeros((1,3))
 tt1=np.zeros((1,3))
 file1=open(fn)
 print ('Extracting data from',fn)
 while j<=jm:
  s='- # '  
  jj=str(j) 
  s=s+jj+'\n'   
  k=1 
  for l in file1:
   if l.find('  weight: ')+1:
    w=l.split(':')[1] 
    w=w.replace("\n","") 
   if l.find(s)+1: 
#    fe=file1.next() 
    fe=next(file1)
    fe=fe.split(':')[1] 
    k=k+1
    if ff==0:
     tt[0][0]=int(j)
     tt[0][1]=int(w)
     tt[0][2]=float(fe)
    else:
     tt1[0][0]=int(j)
     tt1[0][1]=int(w)
     tt1[0][2]=float(fe)
     tt=np.row_stack((tt, tt1))
    ff=ff+1
  file1.seek(0)
  j=j+1
 file1.close()
 return tt
#------------------------------------------------------
t0=extract(fn0,B)
t1=extract(fn1,B)
t2=extract(fn2,B)
f0=t0[:,2]*1e12*2*pi
f1=t1[:,2]*1e12*2*pi
f2=t2[:,2]*1e12*2*pi
w=t0[:,1]
Tem=range(1,Tm,1) 
X=len(Tem)
i=0
#-----------------------------------------------
if An=='n':
 tec=np.zeros((1,X))
 MG=np.ones((1,X))
 if Dm==2:
  print ('2D isotropic material and only 1 independent lattice constant, please check')
  print ('...Calculating...')
  ec=np.array([[c11,c12],[c12,c11]])*1e9
  sc=LA.inv(ec)
  for T in Tem:
   mg1,cv=macrogrun(f0,f1,f2,w,dd,T,B)
   mg1=mg1/Dm
   MG[0][i]=mg1
   tec1=mg1*(sc[0][0]+sc[0][1])*cv/v0
   tec[0][i]=tec1
   i=i+1
 else:
  print ('3D isotropic material and only 1 independent lattice constant, please check')
  print ('...Calculating...')
  ec=np.array([[c11,c12,c12],[c12,c11,c12],[c12,c12,c11]])*1e9
  sc=LA.inv(ec)
  for T in Tem:
   mg1,cv=macrogrun(f0,f1,f2,w,dd,T,B)
   mg1=mg1/Dm
   MG[0][i]=mg1
   tec1=float(mg1*(sc[0][0]+sc[0][1]+sc[0][2])*cv/v0)
   tec[0][i]=tec1
   i=i+1
else:
 t3=extract(fn3,B)
 t4=extract(fn4,B)
 f3=t3[:,2]*1e12*2*pi
 f4=t4[:,2]*1e12*2*pi
 if Dm==2:
  print ('2D anisotropic material and 2 independent lattice constants, please check')
  print ('...Calculating...')
  tec=np.zeros((2,X))
  MG=np.ones((2,X))
  ec=np.array([[c11,c12],[c12,c22]])*1e9
  sc=LA.inv(ec)
  for T in Tem:
   mg1,cv=macrogrun(f0,f1,f2,w,dd,T,B)
   mg2,cv=macrogrun(f0,f3,f4,w,dd,T,B)
   MG[0][i]=mg1
   MG[1][i]=mg2
   tec1=float((mg1*sc[0][0]+mg2*sc[0][1])*cv/v0)
   tec[0][i]=tec1
   tec2=float((mg1*sc[1][0]+mg2*sc[1][1])*cv/v0)
   tec[1][i]=tec2
   i=i+1
 else:
  if IL==3:
   tec=np.zeros((3,X))
   MG=np.ones((3,X))
   t5=extract(fn5,B)
   t6=extract(fn6,B)
   print ('3D anisotropic material and 3 independent lattice constants, please check')
   print ('...Calculating...')
   f5=t5[:,2]*1e12*2*pi
   f6=t6[:,2]*1e12*2*pi
   ec=np.array([[c11,c12,c13],[c12,c22,c23],[c13,c23,c33]])*1e9
   sc=LA.inv(ec)
   for T in Tem:
    mg1,cv=macrogrun(f0,f1,f2,w,dd,T,B)
    mg2,cv=macrogrun(f0,f3,f4,w,dd,T,B)
    mg3,cv=macrogrun(f0,f5,f6,w,dd,T,B)   
    MG[0][i]=mg1
    MG[1][i]=mg2
    MG[2][i]=mg3   
    tec1=float((mg1*sc[0][0]+mg2*sc[0][1]+mg3*sc[0][2])*cv/v0)
    tec[0][i]=tec1
    tec2=float((mg1*sc[1][0]+mg2*sc[1][1]+mg3*sc[1][2])*cv/v0)
    tec[1][i]=tec2
    tec3=float((mg1*sc[2][0]+mg2*sc[2][1]+mg3*sc[2][2])*cv/v0)
    tec[2][i]=tec3
    i=i+1  
  else:
   print ('3D anisotropic material and 2 independent lattice constants, please check')
   print ('...Calculating...')
   tec=np.zeros((2,X))
   MG=np.ones((2,X))
   ec=np.array([[c11,c12,c13],[c12,c22,c23],[c13,c23,c33]])*1e9
   sc=LA.inv(ec)
   for T in Tem:
    mg1,cv=macrogrun(f0,f1,f2,w,dd,T,B)
    mg2,cv=macrogrun(f0,f3,f4,w,dd,T,B)
    mg1=mg1/2
    MG[0][i]=mg1
    MG[1][i]=mg2
    tec1=float((mg1*sc[0][0]+mg1*sc[0][1]+mg2*sc[0][2])*cv/v0)
    tec[0][i]=tec1
    tec2=float((mg1*sc[2][0]+mg1*sc[2][1]+mg2*sc[2][2])*cv/v0)
    tec[1][i]=tec2
    i=i+1  
ltec=np.row_stack((Tem,MG,tec))
ltec=ltec.T
#---------------------------------------------------
Y=ltec.shape[1]
filename='LTEC.dat'
fd=open(filename, 'w')
s0='{:<6}'.format('T (K)')
fd.write(s0)
YY=int((Y-1)/2)
#print(YY)
for j in range(0,YY):
 s0='{:^25}'.format('Macro Gruneisen')
 fd.write(s0)
for j in range(0,YY):
 s0='{:^25}'.format('LTEC (K-1)')
 fd.write(s0)
fd.write('\n')
for i in range(0,X):
 s1=str(int(ltec[i][0]))
 s1='{:<6}'.format(s1)
 fd.write(s1)
 s2=(ltec[i][1])
 s2='{: 20.15e}'.format(s2)
 fd.write(s2)
 for j in range(2,Y):
  s3=ltec[i][j]
  s3='{: 25.15e}'.format(s3)
  fd.write(s3)
 fd.write('\n')
fd.close() 
print ('-----------Successful!------------')

