import numpy as np
import illustris_python as il
import scipy.stats as sts
from scipy.spatial import KDTree

basePath = './data'
host_fields = ['Group_M_Crit200','GroupFirstSub','GroupMass','GroupMassType','GroupPos','Group_R_Crit200','GroupVel'] # list of Group fields to read and set as attributes of hst class
sub_fields = ['SubhaloFlag','SubhaloMass','SubhaloMassInRadType','SubhaloGrNr','SubhaloPos','SubhaloVel','SubhaloStellarPhotometrics','SubhaloSFRinRad'] # list of Suhalo fields to read and set as attributes of sub class
h0 = 0.677
mconv = 10 - np.log10(h0)

class obj(object): # define a class and set class attributes from the keys of dictionary d
    def __init__(self, d):
        for key in d:   setattr(self, key, d[key])
        
class sim(obj): # read the Group and Subhalo catalogs and create the hst and sub objects respectively alongwith their attributes
    def __init__(self,exsub_fields=0,exgrp_fields=0):
            if exgrp_fields: host_fields.extend(exgrp_fields)
            host_halos = il.groupcat.loadHalos(basePath,99,fields=host_fields)
            self.hst = obj(host_halos)
            
            if exsub_fields: sub_fields.extend(exsub_fields)
            sub_halos = il.groupcat.loadSubhalos(basePath,99,fields=sub_fields)
            self.sub = obj(sub_halos)

    def mass_add(self): # take logarithm of mass and SFR values and substitute for Hubble factor h
        setattr(self.hst, 'group_m200', mconv + np.log10(self.hst.Group_M_Crit200))
        setattr(self.hst,'group_mst', mconv + np.log10(self.hst.GroupMassType[:,4]))
        setattr(self.sub,'mgas', mconv + np.log10(self.sub.SubhaloMassInRadType[:,0]))
        setattr(self.sub,'mst',mconv + np.log10(self.sub.SubhaloMassInRadType[:,4]))
        setattr(self.sub,'ssfr',np.log10(1e-15+self.sub.SubhaloSFRinRad)-self.sub.mst)

    

    def gal_select(self):
            massive_gal,dwarf_gal = dwf_select(self.sub.mst)

            dhost = []
            llg,llt,llh,llm = [],[],[],[]
            
            massive_gal_tree = KDTree(self.sub.SubhaloPos[massive_gal,:]) 
            dwarf_host,host_mem = np.unique(self.sub.SubhaloGrNr[dwarf_gal],return_inverse=True)
            Ndh = len(dwarf_host)
            print('Ndh',Ndh)
            for i in range(Ndh):  
                lh = dwarf_host[i]
                if self.hst.group_m200[lh]<11.5 and self.hst.group_m200[lh]>=10:
                    find_mem = dwarf_gal[np.where(host_mem==i)[0]]
                    lt,lg= [],[]
                    dh = []
                    for j in find_mem:
                        dist, ind = massive_gal_tree.query(self.sub.SubhaloPos[j,:], k=6)
                        ind = massive_gal[ind]
                        dist *= 1e-3/h0
                        MD_mass = mconv + np.log10(self.sub.SubhaloMass[ind[1:]])
                
                        lt.append(max(MD_mass - 3*np.log10(dist[1:])) - 10.96)
                        dh.append(dist[1])

                    stellar_sort = np.argsort(self.sub.mst[find_mem])
                    cen = find_mem[stellar_sort[-1]] 
                    
                    r = wrap_dist_1n(self.sub.SubhaloPos[cen,:],self.hst.GroupPos[lh,:])
                    if r<3:
                        dhost.append(dh)
                        llt.append(lt)
                        llh.append(lh)
                        llg.append(find_mem[stellar_sort])
                        llm.append(len(find_mem))
                    else: continue
            
            return llt,np.array(llh),llg,np.array(llm),dhost


    def field_select(self):
            massive_gal,dwarf_gal = dwf_select(self.sub.mst)
            back_gal = backspl_load()

            dhost = [[],[],[]]
            llg,llt,llh = [[],[],[]],[[],[],[]],[[],[],[]]
            
            massive_gal_tree = KDTree(self.sub.SubhaloPos[massive_gal,:]) 

            for j in back_gal:
                lh = self.sub.SubhaloGrNr[j]
                if self.hst.group_m200[lh]<11.5:
                        dist, ind = massive_gal_tree.query(self.sub.SubhaloPos[j,:], k=6)
                        ind = massive_gal[ind]
                        dist *= 1e-3/h0
                        MD_mass = mconv + np.log10(self.sub.SubhaloMass[ind[1:]])
        
                        dhost[2].append(dist[1])
                        llt[2].append(max(MD_mass - 3*np.log10(dist[1:])) - 10.96)
                        llh[2].append(lh)
                        llg[2].append(j)

                dwarf_gal = np.delete(dwarf_gal,np.where(dwarf_gal==j)[0])
                
                
            dwarf_host,host_mem = np.unique(self.sub.SubhaloGrNr[dwarf_gal],return_inverse=True)
            Ndh = len(dwarf_host)
            print('Ndh',Ndh)
            for i in range(Ndh):  
                lh = dwarf_host[i]
                if self.hst.group_m200[lh]<11.5 and self.hst.group_m200[lh]>=10:
                    find_mem = dwarf_gal[np.where(host_mem==i)[0]]

                    stellar_sort = np.argsort(self.sub.mst[find_mem])
                    cen = find_mem[stellar_sort[-1]]

                    r = wrap_dist_1n(self.sub.SubhaloPos[cen,:],self.hst.GroupPos[lh,:])
                    if r<3 and len(find_mem)>=1:
                        dist, ind = massive_gal_tree.query(self.sub.SubhaloPos[cen,:], k=6)
                        ind = massive_gal[ind]
                        dist *= 1e-3/h0
                        MD_mass = mconv + np.log10(self.sub.SubhaloMass[ind[1:]])
            
                        dhost[0].append(dist[1])
                        llt[0].append(max(MD_mass - 3*np.log10(dist[1:])) - 10.96)
                        llh[0].append(lh)
                        llg[0].append(cen)
                        
                        if len(find_mem)>1:
                            for j in find_mem[stellar_sort[:-1]]:
                                dist, ind = massive_gal_tree.query(self.sub.SubhaloPos[j,:], k=6)
                                ind = massive_gal[ind]
                                dist *= 1e-3/h0
                                MD_mass = mconv + np.log10(self.sub.SubhaloMass[ind[1:]])
                
                                dhost[1].append(dist[1])
                                llt[1].append(max(MD_mass - 3*np.log10(dist[1:])) - 10.96)
                                llh[1].append(lh)
                                llg[1].append(j)
                
            return llt,llh,llg,dhost

def backspl_load():
    #back_host = np.genfromtxt('backsplash.txt',usecols=0,dtype='int')
    back_gal = np.genfromtxt('backsplash.txt',usecols=2,dtype='int')
    #back_dist = np.genfromtxt('backsplash.txt',usecols=2)
    
    Nbsg = len(back_gal)
    print(Nbsg)
    return back_gal

def dwf_select(mst,thresh=9.5,floor=7.5):
            massive_gal = np.where(mst>=thresh)[0] 
            Nmass = len(massive_gal)
            
            dwarf_gal = np.where((mst>=floor)&(mst<thresh))[0]
            Ngal = len(dwarf_gal)
            
            print(Nmass,Ngal)
            return massive_gal,dwarf_gal 


def dwf_sfms(mst,ssfr):
        massive_gal,dwarf_gal = dwf_select(mst)

        bw=0.2
        bin_stellar = np.arange(7.5,9.5,bw)
        
        dg1 = dwarf_gal[np.where(ssfr[dwarf_gal]>-15)[0]]
        bin_med, bin_edg, bin_n = sts.binned_statistic(mst[dg1],ssfr[dg1], statistic='median', bins=bin_stellar)
        
        res = sts.linregress(bin_edg[:-1],bin_med)
        print(res.intercept,res.slope)
        #sfr_seq = res.intercept + res.slope*bin_stellar
        return res.intercept,res.slope

def wrap_dist(pos1,pos2):
    dist = np.zeros(len(pos1[:,0]))
    for j in range(3):
        dx = np.abs(pos1[:,j] - pos2[:,j])
        flip = np.where(dx > 17500)[0]
        dx[flip] = 35000 - dx[flip]
        dist += np.square(dx)
    return np.sqrt(dist)

def wrap_dist_1n(pos1,pos2):
    dist = 0
    for j in range(3):
        dx = np.abs(pos1[j] - pos2[j])
        if dx > 17500: dx = 35000 - dx
        dist += np.square(dx)
    return np.sqrt(dist)    


