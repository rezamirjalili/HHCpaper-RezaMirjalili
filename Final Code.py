#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
from tsp_solver.greedy import solve_tsp
from docplex.mp.model import Model
import heapq
import timeit
import itertools 
import operator


# In[ ]:


N = [i for i in range(24)]
Inc = np.array(pd.read_excel('C:\data.xlsx',sheet_name=0),int)
cost = np.array(pd.read_excel('C:\data.xlsx',sheet_name=1),int)
tm = np.zeros((len(N),len(N)),float)
A=[(i,j,cost[i,j]) for i in range(24) for j in range(24) if Inc[i,j]>0 ]
G = nx.Graph()
G.add_weighted_edges_from(A)
for i in N:
    for j in N:
        if i<j:
            tm[i,j]= nx.shortest_path_length(G,i,j,'weight')
            tm[j,i]= tm[i,j]

J = [8,11,19]
pickle.dump( tm, open( "tm_sioux.p", "wb" ) )
pickle.dump(Inc, open( "I_sioux.p", "wb" ) )
pickle.dump(cost, open( "cost_sioux.p", "wb" ) )
pickle.dump(J, open( "J_sioux.p", "wb" ) )


# In[ ]:


N = [i for i in range(526)]
Inc = np.array(pd.read_excel('C:\data.xlsx',sheet_name=4),int)
cost = np.array(pd.read_excel('C:\data.xlsx',sheet_name=5),int)
A=[(i,j,cost[i,j]) for i in range(526) for j in range(526) if Inc[i,j]>0 ]
G = nx.Graph()
G.add_weighted_edges_from(A)
tm = np.zeros((len(N),len(N)),float)
G = nx.Graph()
G.add_weighted_edges_from(A)
for i in N:
    for j in N:
        if i<j:
            tm[i,j]= nx.shortest_path_length(G,i,j,'weight')
            tm[j,i]= tm[i,j]
pickle.dump( tm, open( "tm_chic.p", "wb" ) )
pickle.dump(Inc, open( "I_chic.p", "wb" ) )
pickle.dump(cost, open( "cost_chic.p", "wb" ) )
pickle.dump(J, open( "J_chic.p", "wb" ) )


# # Sioux Falls data load

# In[63]:


tm = pickle.load( open( "tm_sioux.p", "rb" ) )
Inc = pickle.load( open( "I_sioux.p", "rb" ) )
cost = pickle.load( open( "cost_sioux.p", "rb" ) )
J = pickle.load( open( "J_sioux.p", "rb" ) )
n = 24


# # Load Chicago Data

# In[219]:


tm = pickle.load( open( "tm.p", "rb" ) )
Inc = pickle.load( open( "I.p", "rb" ) )
cost = pickle.load( open( "cost.p", "rb" ) )
J = [400, 369,  99, 125, 184,  48, 171, 205, 210, 218,  89, 279, 307,
       327, 340, 354, 382, 440, 455, 486]#J = pickle.load( open( "J.p", "rb" ) )
n = 526
A=[(i,j,cost[i,j]) for i in range(526) for j in range(526) if Inc[i,j]>0 ]
G = nx.Graph()
G.add_weighted_edges_from(A)


# In[220]:


N = [i for i in range(n)]
I = [i for i in N if i not in J]
u_e = {i:min([tm[i,j] for j in J]) if i in I else 0 for i in N}
u_l = {i:2*max([tm[i,j] for j in J]) if i in I else 0 for i in N}

FC={j:random.randint(20,40)*10000 for j in J}
sm = {i: random.choice([j*5 for j in range(1,7)]) if i in I else 0 for i in N}# Mild condition 
ss = {i: random.choice([j*5 for j in range(1,5)]) if i in I else 0 for i in N}# Severe Condition
mu = {i:(random.choice([1,2,3])*250)**-1 if i in I else 0  for i in N}
def ct(i,j,r):
    M = sum([mu[k] for k in r if k not in J])
    #print(M)
    d=0
    for k in r:
        d += ((tm[i,k]+tm[k,j]+tm[i,j])*0.5+ss[k]+tm[k,j])*mu[k]/M
    return round((np.exp(-u_e[i]*M)-np.exp(-u_l[j]*M))*d*M)
cap = 10000    


# In[225]:


B = 10**7


# In[222]:


def create_data_model(I,j):
    """Stores the data for the problem."""
    #tm = np.array(pd.read_excel('C:\data2.xlsx',sheet_name=0),int)
    #tm.astype(np.int64)
    #tw = np.array(pd.read_excel('C:\data2.xlsx',sheet_name=1),int)
    #tw.astype(np.int64)
    N = [j]+I
    data = {}
    data['time_matrix'] = [[int(tm[i,jj]+ct(i,jj,N)+sm[i]) for jj in N] for i in N]
    data['time_windows'] = [(int(u_e[i]),int(u_l[i])) for i in N]
    data['num_vehicles'] = 20
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    p = []
    for vehicle_id in range(data['num_vehicles']):
        p.append([])
        index = routing.Start(vehicle_id)
        p[-1].append(index)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            index = assignment.Value(routing.NextVar(index))
            p[-1].append(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(
            manager.IndexToNode(index), assignment.Min(time_var),
            assignment.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            assignment.Min(time_var))
        #print(plan_output)
        total_time += assignment.Min(time_var)
    #print('Total time of all routes: {}min'.format(total_time))
    return p
def VRP(I,j,cap):
    """Solve the VRP with time windows."""
    data = create_data_model(I,j)
    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30,  # allow waiting time
        cap,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        #print(time_window)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    assignment = routing.SolveWithParameters(search_parameters)
    N = [j]+I
    if assignment:
        p= print_solution(data, manager, routing, assignment)
        for i in range(len(p)):
            if len(p[i])<=2:
                p[i]='f'
            else:
                p[i][0]=0
                p[i][-1]=0
        while 'f' in p:
            p.remove('f')
        path= []
        for i in p:
            path.append([])
            for jj in i:
                path[-1].append(N[jj])
        for i in range(len(path)):
            path[i]=path[i][1:len(path[i])-1]
            
        return path
    else:
        return []
def cof(ix,tb):
    if ix in tb:
        return 10**20
    else:
        return 1


# In[232]:


tmr = 0
strt = timeit.default_timer()
C = initial_routes()
H=nx.DiGraph()
routes = {}
for j in C:
    if C[j]!=[]:
        nr = VRP(C[j],j,cap)
            
        routes= update_dic(routes,j,nr)
        
r=[(j,k) for j,k in routes]
tw={(j,k):tw_cal(routes[j,k],j,0) for j,k in r}
tb = []
UB,s = IRL(routes,tw,r)
lnd,sl,L= LR(routes,tw,r,UB+1)
DZ = [UB]
l = DLP2(lnd,tw,r)
m=[1]
JJ = sl[0]
tmr+=timeit.default_timer()-strt


# In[240]:


#Extend service
strt = timeit.default_timer()
j0=1
while j0!='f':
    j0,j1,rt = extend_service(JJ,sl[1])
    if j0!= 'f':
        m =[]
        for j,k in sl[1]:
            if j in [j0,j1]:
                m.append((j,k))
        routes = update_dic(routes,j1,rt)
        r = update_r(r,routes)
        tw = update_tw(tw,routes)
        tb+=m
        lnd,sl,L=LR(routes,tw,r,UB)
        print(round(L),len(sl[0]))
        DZ.append(round(L))
        #UB = L+1
        JJ = sl[0]
        l = DLP2(lnd,tw,r)

tmr+=timeit.default_timer()-strt


# In[241]:





# In[242]:


strt = timeit.default_timer()
br=1
while br!='f':
    br = swap_service(JJ,sl[1])
    if br!= 'f':
        m =[]
        for j1,j2 in br:
            for j,k in sl[1]:
                if j==j1:
                    m.append((j,k))
            routes = update_dic(routes,j2,br[j1,j2])
        r = update_r(r,routes)
        tw = update_tw(tw,routes)
        tb+=m
        lnd,sl,L=LR(routes,tw,r,UB)
        print(round(L))
        DZ.append(round(L))
        #UB = L+1
        JJ = sl[0]
        l = DLP2(lnd,tw,r)

tmr+=timeit.default_timer()-strt

#AS['length'].update({cap:{'CpuTime':tmr,'#iteration':len(DZ),'Z':DZ[-1],'#Vehicles':len(sl[1]),'Max length':max([tw[j,k] for j,k in sl[1]])}})


# In[243]:


strt = timeit.default_timer()
chk = 1
while chk>0 and len(sl[0])>1:
    imp={}
    rt ={}
    #print('exchange has started')
    for j1,k1 in sl[1]:
        for j2,k2 in sl[1]:
            if j1!=j2 and (j2,k2,j1,k1) not in imp:
                X,Y,z= update_current_routes(j1,k1,j2,k2)
                if z!=0:
                    print((j1,k1,j2,k2),z)
                    imp.update({(j1,k1,j2,k2):z})
                    rt.update({(j1,k1,j2,k2):[X,Y]})
    if len(imp)==0:
        chk = 0
    else: 
        while len(imp)>0:
            j1,k1,j2,k2 = min_dict(imp)
            tb+=[(j1,k1),(j2,k2)]
            if len(rt[j1,k1,j2,k2][0])>0:
                routes = update_dic(routes,j1,rt[j1,k1,j2,k2][0])
            if len(rt[j1,k1,j2,k2][1])>0:
                routes = update_dic(routes,j2,rt[j1,k1,j2,k2][1])
            for jj1,kk1,jj2,kk2 in imp:
                if (jj1==j1 and kk1==k1) or (jj2==j1 and kk2==k1):
                    imp[jj1,kk1,jj2,kk2]='f' 
                if (jj2==j2 and kk2==k2) or (jj1==j2 and kk1==k2):
                    imp[jj1,kk1,jj2,kk2]='f'
            imp = {(j1,k1,j2,k2): imp[j1,k1,j2,k2] for j1,k1,j2,k2 in imp if imp[j1,k1,j2,k2]!='f'}
        r = update_r(r,routes)
        tw = update_tw(tw,routes)
        lnd,sl,L=LR(routes,tw,r,UB)
        print(round(L))
        DZ.append(round(L))
        JJ = sl[0]
tmr+=timeit.default_timer()-strt


# In[236]:


DZ


# In[248]:


tmr


# In[245]:


cap


# In[161]:


AS


# In[118]:





# In[231]:


def extend_service(JJ,active_route):
    #Close a center by adding its nodes to anothers center
    sf = {j:Gama*sum([tw[jj,k] for jj,k in active_route if jj==j])/FC[j] for j in JJ}
    sf = {j:sf[j] for j in JJ if sf[j]<=1}
    rk = {rank: key for rank, key in enumerate(sorted(sf, key=sf.get, reverse=False), 1)}
    a = rk[1]
    b = rk[len(rk)]
    if a==b:
        return 'f','f','f'
    rt = {a:[],b:[]}
    t={a:0,b:0}
    for j,k in active_route:
        if j==a or j==b :
            rt[j]+=routes[j,k]
            t[j]+=tw[j,k]
    check =1 
    while check>0:
        nr = VRP(rt[b]+rt[a],b,cap)
        x = tw_cal(nr,b,1)*Gama-(t[a]+t[b])*Gama-FC[a]
        if x<0:
            print(x)
            return a,b, nr 
        else:
            del rk[len(rk)]
            print(len(rk))
            b = rk[len(rk)]
            rt.update({b:[]})
            t.update({b:0})
            if a==b:
                return 'f','f','f'
            for j,k in active_route:
                if j==b :
                    rt[j]+=routes[j,k]
                    t[j]+=tw[j,k]
    return 'f','f','f'

def swap_service(JJ,active_route):
    #assign all the routes of an open station to the closed one
    nJ = [j for j in J if j not in JJ]
    rj={j:[] for j in JJ}
    c = {j:FC[j] for j in JJ}
    for j,k in active_route:
        rj[j]+=routes[j,k]
        c[j]+=Gama*tw[j,k]
    nc = {}
    nr = {}
    for j in JJ:
        d1 = 1#sum([tm[i,j] for i in rj[j]])
        for jj in nJ:
            d2 = 0#sum([tm[i,jj] for i in rj[j]])
            if d2<d1:
                print(j,jj)
                nr[j,jj] = VRP(rj[j],jj,cap)
                nc[j,jj] = FC[jj]+Gama*tw_cal(nr[j,jj],jj,1)-c[j]

    a = min_dict(nc)
    bc={}
    br={}
    if nc[a]>=0:
        return 'f'
    while len(nc)>0:
        bc[a] = nc[a]
        br[a] = nr[a]
        j1,j2 = a
        for j,jj in nc:
            if j1 in [j,jj] or j2 in [j,jj]:
                nc[j,jj] = 'f'
                nr[j,jj] = 'f'
        nc = {(j,jj):nc[j,jj] for j,jj in nc if nc[j,jj]!='f'}
        nr = {(j,jj):nr[j,jj] for j,jj in nr if nr[j,jj]!='f'}
        if len(nc)==0:
             return br
        a = min_dict(nc)
        if nc[a]>=0:
            break
    return br
            


# In[67]:





# In[62]:


len(sl[0])


# In[190]:


def update_current_routes(j1,k1,j2,k2):
    X = [i for i in routes[j1,k1]]
    Y = [j for j in routes[j2,k2]]
    A = [i for i in X]
    B = [j for j in Y]
    oz = (tw[j1,k1]+tw[j2,k2])
    check =1
    Best=[]
    bZ = []
    while check==1:
        dz ={}
        br1 = broken(j1,A,j2,B)
        br2 = broken(j2,B,j1,A)
        if br1==[] and br2==[]:
            check=0
        for i in br1:
            Ai = [k for k in A if k!=i]
            a = 0
            if len(Ai)>0:
                Ari = VRP(Ai,j1,cap)
                a =tw_cal(Ari,j1,1)
            Bi = [k for k in B]+[i]
            Bri = VRP(Bi,j2,cap)
            a+=tw_cal(Bri,j2,1)#red_cost(Aj,l,j1,JJ)+red_cost(Bj,l,j2,JJ)
            if a<oz:
                dz[i,'r']=a
        for j in br2:
            Aj = [k for k in A]+[j]
            Arj = VRP(Aj,j1,cap)
            Bj = [k for k in B if k!=j]
            b = tw_cal(Arj,j1,1)
            if len(Bj)>0:
                Brj = VRP(Bj,j2,cap)
                b+=tw_cal(Brj,j2,1)#red_cost(Bi,l,j2,JJ)+red_cost(Ai,l,j1,JJ)
            if b<oz:
                dz[j,'g']=b          
        if len(dz)>0:
            idz  =min_dict(dz)
            if idz[1]=='g':
                A.append(idz[0])
                B.remove(idz[0])
            elif idz[1]=='r':
                A.remove(idz[0])
                B.append(idz[0])
            oz = dz[idz]
            Best.append([A,B])
            bZ.append(oz)
        else:
            check = 0
    
    if len(bZ)>0:
        a = bZ.index(min(bZ))
        if len(Best[a][0])>0:
            nX =VRP(Best[a][0],j1,cap)
        else:
            nX =[]
        if len(Best[a][1])>0:
            nY =VRP(Best[a][1],j2,cap)
        else:
            nY = []
        return nX,nY,a-(tw[j1,k1]+tw[j2,k2])
    else:
        return [],[],0
    
def broken(j1,X,j2,Y):
    ry = [j2]+Y+[j2]
    rx = [j1]+X+[j1]
    br={}
    for i in X:
        u = rx.index(i)
        dx = tm[rx[u-1],rx[u+1]]-tm[rx[u-1],rx[u]]-tm[rx[u],rx[u+1]]
        for j in range(len(ry)-1):
            (a,b)=ry[j],ry[j+1]
            dy = tm[a,i]+tm[i,b]-tm[a,b]
            if dy+dx<0:
                br[i,j] = dy+dx
    if len(br)>0:
        return list(set([i for i,j in br]))

    else:
        return []
            
def self_improvement(sl):
    rt = {j:[] for j in sl[0]}
    for jj in  sl[0]:
        jI= []
        zj = 0
        for j,k in sl[1]:
            if j==jj:
                jI+=routes[j,k]
                zj+=tw[j,k]
        nr=VRP(jI,j,cap)
        nz = tw_cal(nr,j,1)
        rt[j]+=nr
    return rt
        


# In[186]:


tw


# In[8]:


def IRL(routes,tw,r):
    #J = list(set([j for j,k in r]))
    md = Model('IRL')
    x = md.binary_var_dict(J,name='x')
    z = md.binary_var_dict(r,name='z')
    obj = md.sum(FC[j]*x[j] for j in J)+md.sum((Gama*tw[j,k])*z[j,k] for j,k in r)
    md.minimize(obj)
    md.add_constraints(md.sum(alpha(routes[j,k],i)*z[j,k] for j,k in r)>=1 for i in I)
    md.add_constraints(z[j,k]-x[j]<=0 for j,k in r)
    md.add_constraint(sum([FC[j]*x[j] for j in J])<=B)
    sol = md.solve(log_output=False)
    solution = [[j for j in J if x[j].solution_value>0],[(j,k) for (j,k) in r if z[j,k].solution_value>0]]
    objfun =md.solution.get_objective_value()
    return objfun,solution

def LR_optimality(cs):
    for j,k in cs:
        if cs[j,k]>0:
            return False
    return True

def tw_cal(r,j,d):
    if d==0:# it is just a route
        a = length_of_route(r,j)
        return a
    else:# it contains more than 1 route
        t = []
        for i in r:
            a = length_of_route(i,j)
            t.append(a)
        return sum(t)

def update_child(mt,routes,r,mother):
    for j,k in routes:
        if (j,k) not in r:
            mt.update({(j,k): mother})
    return mt

def LR(routes,tw,r,UB):
    lnd = {(j,k):0 for j,k in r}
    L,s,cs = MPIRL(routes,tw,r,lnd)
    LB=[0,L]
    while not LR_optimality(cs):

        eps = (UB-LB[-1])/sum([cs[j,k]**2 for j,k in cs])
        lnd = {(j,k): max(0,lnd[j,k]+eps*cs[j,k]) if cs[j,k]>0 else lnd[j,k] for j,k in cs} 
        L,s,cs = MPIRL(routes,tw,r,lnd)
        LB.append(L)
    return lnd,s,L

def MP1(lnd):
    md = Model('MP1')
    x = md.continuous_var_dict(J,name='x')
    md.add_constraint(md.sum(FC[j]*x[j] for j in J)<=B)
    ld = lnd_calc(lnd)
    md.add_constraints(x[j]<=1 for j in J)
    md.minimize(md.sum((FC[j]-ld[j])*x[j] for j in J))
    md.solve(log_output=False)
    return [j for j in J if x[j].solution_value==1]

def MP2(routes,I,tw,r,lnd):
    md = Model('MP2')
    z = md.continuous_var_dict(r,name='z')
    md.minimize(md.sum((Gama*tw[j,k]*cof((j,k),tb)+lnd[j,k])*z[j,k] for j,k in r))
    md.add_constraints(md.sum(alpha(routes[j,k],i)*z[j,k] for j,k in r)>=1 for i in I)
    sol = md.solve(log_output=False)
    s = {(j,k): z[j,k].solution_value for (j,k) in r}
    return s

def DLP1(lnd):
    md=Model('DLP1')
    u = md.continuous_var(name = 'u')
    v = md.continuous_var_dict(J,name='v')
    md.minimize(B*u+md.sum(v[j] for j in J))
    ld = lnd_calc(lnd)
    md.add_constraints(ld[j]<=FC[j]*(u+1)+v[j] for j in J)
    md.solve(log_output=False)
    return u.solution_value,{j:v[j].solution_value for j in J}

def DLP2(lnd,tw,r):
    md = Model('DLP2')
    l = md.continuous_var_dict(I,name='l')
    md.add_constraints(md.sum(alpha(routes[j,k],i)*l[i] for i in I)-Gama*tw[j,k]*cof((j,k),tb)-lnd[j,k]<=0 for j,k in r if (j,k) not in tb)
    md.maximize(md.sum(l[i] for i in I))
    md.solve(log_output=False)
    return {i:l[i].solution_value for i in I}

def alpha(r,x):
    if x in r:
        return 1
    else:
        return 0


def lnd_calc(lnd):
    ld={}
    for j in J:
        a = 0
        for jj,k in lnd:
            if jj==j:
                a+=lnd[j,k]
        ld.update({j:a})
    return ld
    
def MPIRL(routes,tw,r,lnd):
    md = Model('MPIRL')
    x = md.continuous_var_dict(J,name='x')
    z = md.continuous_var_dict(r,name='z')
    ld = lnd_calc(lnd)
    obj = md.sum((FC[j]-ld[j])*x[j] for j in J)+md.sum((Gama*tw[j,k]*cof((j,k),tb)+lnd[j,k])*z[j,k] for j,k in r)
    
    md.add_constraints(md.sum(alpha(routes[j,k],i)*z[j,k] for j,k in r)>=1 for i in I)
    md.add_constraint(md.sum(FC[j]*x[j] for j in J)<=B)
    md.add_constraints(x[j]<=1 for j in J)
    
    md.minimize(obj)
    
    sol = md.solve(log_output=False)
    solution = [[j for j in J if x[j].solution_value>0 ],[(j,k) for (j,k) in r if z[j,k].solution_value>0]]

    cs={}
    for j,k in r:
        d = z[j,k].solution_value-x[j].solution_value
        ##print(d)
        cs.update({(j,k):d})
    #print([(j,k) for j,k in cs if cs[j,k]>0])
    objfun =md.solution.get_objective_value()#+sum([-cs[j]*lnd[j] for j in J if cs[j]<=0])  
    ##print(objfun)
    return objfun,solution,cs




def length_of_route(r,j):
    if len(r)>0:
        d = tm[j,r[0]]+tm[r[-1],j]
        for i in range(len(r)-1):
            d+=tm[r[i],r[i+1]]
        return d
    else:
        d=0
        return 0

def red_cost(r,l,j,JJ):
    if r!=[]:
        t = tw_cal(r,j)
        if j in JJ:
            D=Gama*t-sum([alpha(r,m)*l[m] for m in I])
        else:
            D=FC[j]-sum([alpha(r,m)*l[m] for m in I])+Gama*t
    else:
        D = 0
    return D



def new_always_good(JJ,r,j,lnd):
    B={}
    rd = {}
    for jj in JJ:
        if j!=jj:
            X = []
            Y = []
            for i in r:
                if tm[i,j]<tm[i,jj]:
                    X.append(i)#new route for station j
                else:
                    Y.append(i)#new route for station jj
            if X!=[] and Y!=[]:
                a =red_cost(X,l,j,JJ)+red_cost(Y,l,jj,JJ)
                if a<0:
                    B.update({jj:[X,Y]})
                    rd.update({jj:a})
            elif Y!=[] and X==[]:
                a =red_cost(Y,l,jj,JJ)
                if a<0:
                    B.update({jj:[X,Y]})
                    rd.update({jj:a})
    if len(rd)>0:
        a = min_dict(rd)
        return a,B[a][0],B[a][1]
    else:
        return 'f',[],[] 

    
def decompose(X,JJ):
    k=2
    #tt = Gama*length_of_route(TSPsolve(X,jj),jj)
    #split the nodes into two cluster
    t = matrix_of_cost(X)
    m= []
    while len(m)==0:
        try:
            m,c = kMedoids(t,k,tmax=10000)
            for i in c:
                if c[i]==[]:
                    m=[]
        except:
            m=[]
    c = [[X[j] for j in c[i]] for i in c]
    #print(c)
    rd = np.zeros((len(J),k),float)
    for j in range(len(J)):
        for i in range(k):
            rd[j,i] = red_cost(c[i],l,J[j],JJ)
    ##print(rd)
    dest =[]
    for i in range(k):
        a = list(rd[:,i])
        dest.append(a.index(min(a)))
    x=0
    for j in range(len(dest)):
        x+=rd[dest[j],j]
    if x<0:
        #print(x)
        dest=[J[j] for j in dest]
        #c[0],c[1],z = update_current_routes(c[0],c[1],dest[0],dest[1])
        return {dest[0]:c[0]},{dest[1]:c[1]}
    else:
        return 'f','f'



def matrix_of_cost(X):
    t=np.zeros((len(X),len(X)),float)
    for i in range(len(X)):
        for j in range(len(X)):
            if i<j:
                t[i,j]=tm[X[i],X[j]]
                t[j,i]=t[i,j]
    return t
def update_tw(tw,routes):
    for j,k in routes:
        if (j,k) not in tw:
            tw.update({(j,k): tw_cal(routes[j,k],j,0)})
    return tw
def update_r(r,routes):
    for j,k in routes:
        if (j,k) not in r:
            r.append((j,k))
    return r

def update_dic(routes,j,r):
    x = 0
    for jj,k in routes:
        if j==jj:
            x+=1
    for k in range(len(r)):
        routes.update({(j,k+x): r[k]})
    return routes
def min_dict(d):
    xx = np.inf
    for i in d:
        if d[i]<xx:
            xx = d[i]
            yy = i
    return yy 
def max_dict(d):
    xx = -np.inf
    for i in d:
        if d[i]>xx:
            xx = d[i]
            yy = i
    return yy 
def initial_routes():
    fc = {j:FC[j] for j in J}
    f = []
    jj = []
    C={j:[] for j in J}
    while sum(f)<=B and len(fc)>0:
        j=min_dict(fc)
        f.append(fc[j])
        jj.append(j)
        del fc[j]
    if sum(f)>B:
        f=f[:len(f)-1]
        jj=jj[:len(f)-1]
    
    for i in I:
        dis = {j:tm[i,j] for j in jj}
        
        j=min_dict(dis)
        C[j].append(i)
    return C   


# In[9]:


def should_be_added(sl,m,ch):
    jj = list(set([j for j,k in sl[1] if (j,k) not in m ]))
    #print(jj)
    dz = 0
    for j,k in m:
        dz-=tw[j,k]*Gama
        if j not in jj:
            dz-=FC[j]
    for j,k in ch:
        dz+=tw[j,k]*Gama
        if j not in jj:
            jj.append(j)
            dz+=FC[j]
    #print(dz)
    if dz<0:
        return True
    else:
        return False
def num_active_route(active_route,j):
    x = 0
    for jj,k in active_route:
        if jj==j:
            x+=1
    return x

    


# In[42]:


plt.plot([i+1 for i in range(len(DZ))], DZ)
plt.xlabel('Iteration')
plt.ylabel('Objective function')
plt.savefig('DZ_sioux falls-g100.eps', format='eps', dpi=1000)


# In[44]:


def kMedoids(D, k, tmax=1000):
    # determine dimensions of distance matrix D
    m, n = D.shape
    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))
    # create a copy of the array of medoid indices
    Mnew = np.copy(M)
    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
            # update cluster medoids21
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
            np.sort(Mnew)
            # check for convergence
            if np.array_equal(M, Mnew):
                break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    # return results
    return M, C


# In[46]:


MM,CC = kMedoids(tm, 20, tmax=10000)


# In[215]:


MM

