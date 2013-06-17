# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np;
import networkx as nx
import pickle
import os, sys
import scipy.linalg
import matplotlib.pyplot as plt
import igraph

# <codecell>

def nx_to_igraph(nxG):
    a=nx.adjacency_matrix(nxG);
    aa=list(np.array(a));
    g=igraph.Graph.Weighted_Adjacency(aa, attr="weight", loops=True);
    return g;

# <codecell>

def timevarying_graph_generator(data, edgelist_file,dim):
    primal_graph=nx.read_edgelist(edgelist_file);
    edgelist=[];
    for edge in primal_graph.edges():
        edgelist.append([edge[0],edge[1]]);
    edgelist=np.array(edgelist)
    print data.shape, edgelist.shape
    tempG={};
    if dim!=0:
        data=np.transpose(data);
    print data.shape
    T=data.shape[1];
    print T
    nodes=primal_graph.nodes()
    relabel_dict={};
    count=0;
    for node in primal_graph.nodes():
        relabel_dict[node]=count;
        count+=1;
    print 'Node relabeling complete.'
    print 'Beginning construction of time-graph...'
    for t in range(T):
        if t%1000==0:
            print 'tempo', t;
        g=nx.Graph();
        g.add_nodes_from(relabel_dict.values());
        for i,state in enumerate(data[:,t]):
            if state==1:
                try:
                    g.add_edge(relabel_dict[str(int(edgelist[i][0]))], relabel_dict[str(int(edgelist[i][1]))]);
                except:
                    print i,state;
        tempG[t]=[];
        tempG[t]=g;
    return tempG;

# <codecell>

def calculate_transition_matrices(TG,N=None):
    transition_matrix_dict={};
    if N==None:
        N=0;
        for t in TG:
            graph=TG[t];
            if graph.number_of_nodes()>N:
                N=graph.number_of_nodes();
    
    for t in TG:
        graph=TG[t];
        adj=nx.to_numpy_matrix(graph);
        M=np.eye(N)+np.asarray(adj);
        d=np.dot(M,np.ones((N,)));
        for i,e in enumerate(d):
            if e<=0:
                print 'found zero row ', i, e, 'substituting it with 1';
                d[i]=1;
    
        D=np.diag(d,0);

        transition_matrix_dict[t]=[];
        transition_matrix_dict[t]=np.dot(scipy.linalg.inv(D),M);
    return transition_matrix_dict



# <codecell>

def activity_values(TG,m=1):
    N=0;
    for t in TG:
        graph=TG[t];
        if graph.number_of_nodes()>N:
            N=graph.number_of_nodes();
    
    activity={};
    z=[];
    for t in TG:
        graph=TG[t];
        zz=0;
        degree=graph.degree();
        for n in degree:
            if n in activity:
                if degree[n]>0:
                    activity[n]+=1;
                    zz+=1;
            else:
                activity[n]=[];
                if degree[n]>0:
                    activity[n]=1;
                    zz+=1;
                else:
                    activity[n]=0;
        z.append(zz);
        
    norm_activity={};
    eta = np.mean(z)/(np.mean(np.array(activity.values()))*N*m);
    for key in activity:
        norm_activity[key]=[];
        norm_activity[key]=float(eta)*float(activity[key])/float(len(activity));
    return norm_activity;  




def stationary_weights_distributions(activity_dict,m,num_walkers,epsilon=0.1):
    import sympy as sp;
    av_activity=np.mean(activity_dict.values());
    num_walkers=num_walkers/float(len(activity_dict)); 
    x=0;
    num_iter=0;
    step=0.0001;
    phi_precision=100;
    while (phi_precision>epsilon) and (num_iter<1000):
        phi_precision=0;
        for e in activity_dict:
            phi_precision+=activity_dict[e]*(1/float(len(activity_dict))) * (m*num_walkers*activity_dict[e]+x)/(activity_dict[e]+m*av_activity);
        phi_precision=np.abs(x-phi_precision);
        print phi_precision
        x+=step;
        num_iter+=1;
    print x, num_iter    
    w = (np.array(activity_dict.values())*m*num_walkers+x)/ (np.array(activity_dict.values())+m*av_activity);
    
    return w/np.sum(w);

# <codecell>

def project_onto_community(TG, unproj_R, partition):
    projected_R_tau={};
    N=0;
    for t in TG:
        graph=TG[t];
        if graph.number_of_nodes()>N:
            N=graph.number_of_nodes();
    num_comms=len(list(set(partition.values())));        
    partition_index={};
    for i,p in enumerate(list(set(partition.values()))):
        partition_index[p]=i;
    H=np.zeros((N,num_comms));
    print H.shape;
    for node in partition:
        H[node][partition_index[partition[node]]]=1;
    for key in unproj_R:
        projected_R_tau[key]=[];
        projected_R_tau[key] = np.trace(np.dot(np.transpose(H),np.dot(unproj_R[key],H))); 
    return projected_R_tau;

# <codecell>

def aggregate_graph(TG):
    new_graph=nx.Graph();
    for t in TG:
        graph=TG[t];
        for node in graph.nodes():
            if not new_graph.has_node(node):
                new_graph.add_node(node);
        for edge in graph.edges(data=True):
            if new_graph.has_edge(edge[0],edge[1]):
                new_graph[edge[0]][edge[1]]['weight']+=1;
            else:
                new_graph.add_edge(edge[0], edge[1], weight=1);

    return new_graph;


def limited_aggregate_graph(TG,t_min,scale,periodic=True):
    new_graph=nx.Graph();
    tmax=np.max(TG.keys());
    for t in range(t_min,t_min+scale):
        if periodic==True:
            graph=TG[t%tmax];
        else:
            graph=TG[t];
        for node in graph.nodes():
            if not new_graph.has_node(node):
                new_graph.add_node(node);
        for edge in graph.edges(data=True):
            if new_graph.has_edge(edge[0],edge[1]):
                new_graph[edge[0]][edge[1]]['weight']+=1;
            else:
                new_graph.add_edge(edge[0], edge[1], weight=1);

    return new_graph;



# <codecell>

def exponential_weighted_aggregated_temporal_graph(TG,omega=0.1):
    edge_dict={};
    for t in TG:
        for edge in TG[t].edges():
            if edge in edge_dict:
                edge_dict[edge].append(t);
            else:
                edge_dict[edge]=[];
                edge_dict[edge].append(t);
    ExpG=nx.Graph();
    for edge in edge_dict:
        w=np.sum(np.exp(-float(omega)*np.diff(np.array(edge_dict[edge]))));
        ExpG.add_edge(edge[0],edge[1],weight=w);
    return ExpG;

# <codecell>

def instant_partition_matrix(partition,verbose=False):
    partition_index={};
    for i,p in enumerate(list(set(partition.values()))):
        partition_index[p]=i;
    N=len(list(set(partition.keys())));
    num_comms=len(list(set(partition.values())));        
    H=np.zeros((N,num_comms));
    if verbose==True:
        print H.shape;
    for node in partition:
        H[node][partition_index[partition[node]]]=1;
    return H; 


def stationary_matrix(w):
    N=len(w);
    stationary_matrix=np.zeros((N,N));
    for i,n in enumerate(w):
        for j,l in enumerate(w):
            stationary_matrix[i][j]=n*l;
    return stationary_matrix;



def temporal_projected_stationary_stability(TG,M_matrices, w, tau_values, temporal_partition, show_fig=False):
    proj_R_tau={};
    #define at which physical times the partition changes
    partition_temporal_swaps = sorted(temporal_partition.keys());
    #find number of nodes involved
    N=0;
    for t in TG:
        graph=TG[t];
        if graph.number_of_nodes()>N:
            N=graph.number_of_nodes();

    stationary_matrix=np.zeros((N,N));
    for i,n in enumerate(w):
        for j,l in enumerate(w):
            stationary_matrix[i][j]=n*l;
    
    
    #start of loop on markov times 
    for tau in tau_values:
        print tau
        proj_R_tau[tau]=[];
        counter=0;
        tr_av=[];
        keys=M_matrices.keys();
        current_partition_time=0;
        
        H=instant_partition_matrix(temporal_partition[partition_temporal_swaps[current_partition_time]]);
        #start of loop on physical times
        for t in range(len(M_matrices)-tau):
            
            #check whether the partition is still the correct one for the physical time
            if partition_temporal_swaps[(current_partition_time+1)%len(partition_temporal_swaps)]<=t and (current_partition_time+1)<len(partition_temporal_swaps):
                current_partition_time+=1;
                H=instant_partition_matrix(temporal_partition[partition_temporal_swaps[current_partition_time]]);
            
            M_disposable=np.eye(N);
            for i in range(tau):
                M_disposable=np.dot(M_disposable,M_matrices[keys[t+i]]);
            M_disposable=np.dot(np.diag(w,0),M_disposable) - stationary_matrix;
            tr_av.append(np.trace(np.dot(H.T,np.dot(M_disposable,H))));
            del M_disposable;
        del H;
        proj_R_tau[tau] = [];
        proj_R_tau[tau] = np.mean(tr_av);
        del tr_av;
    return proj_R_tau;


def spinglass_partition(TG, t_min, t_max):
    import igraph as ig;
    import os;

    TG_suppl={};
    #select correct section of times
    for t in TG:
        if t>=t_min and t<t_max:
            TG_suppl[t]=[];
            TG_suppl[t]=TG[t];

    new_graph=aggregate_graph(TG_suppl);
    print new_graph.number_of_nodes();

#    nx.write_pajek(new_graph,'temp_graph.net');
#    g=ig.read('temp_graph.net',format="pajek")
#    os.remove('temp_graph.net');
    Q=nx_to_igraph(new_graph).community_spinglass()
    part_q={};
 #   print Q
    for i,mem in enumerate(Q.membership):
            part_q[i]=[];
            part_q[i]=mem;
    return part_q;


def fast_modularity_partition(TG, t_min, t_max):
    import igraph as ig;
    import os;

    #select correct section of times
    TG_suppl={};
    for t in TG:
        if t>=t_min and t<t_max:
            TG_suppl[t]=[];
            TG_suppl[t]=TG[t];

    new_graph=aggregate_graph(TG_suppl);
    #print new_graph.number_of_nodes();
   
#    nx.write_pajek(new_graph,'temp_graph.net');
#    g=ig.read('temp_graph.net',format="pajek")
#    os.remove('temp_graph.net');
    Q=nx_to_igraph(new_graph).community_fastgreedy();
    return Q; 
#     part_q={};
# #    print Q
#     for i,mem in enumerate(Q.membership):
#             part_q[i]=[];
#             part_q[i]=mem;
#     return part_q;


def modularity_partition(TG, t_min, t_max):
    import igraph as ig;
    import os;

    #select correct section of times
    TG_suppl={};
    for t in TG:
        if t>=t_min and t<t_max:
            TG_suppl[t]=[];
            TG_suppl[t]=TG[t];

    new_graph=aggregate_graph(TG_suppl);
    #print new_graph.number_of_nodes();
   
#    nx.write_pajek(new_graph,'temp_graph.net');
#    g=ig.read('temp_graph.net',format="pajek")
#    os.remove('temp_graph.net');
    Q=nx_to_igraph(new_graph).community_optimal_modularity();

    part_q={};
#    print Q
    for i,mem in enumerate(Q.membership):
            part_q[i]=[];
            part_q[i]=mem;
    return part_q;

def infomap_partition(TG, t_min, t_max):
    import igraph as ig;
    import os;

    #select correct section of times
    TG_suppl={};
    for t in TG:
        if t>=t_min and t<t_max:
            TG_suppl[t]=[];
            TG_suppl[t]=TG[t];

    new_graph=aggregate_graph(TG_suppl);
    #print new_graph.number_of_nodes();
#    nx.write_pajek(new_graph,'temp_graph.net');
#    g=ig.read('temp_graph.net',format="pajek")
#    os.remove('temp_graph.net');
    Q=nx_to_igraph(new_graph).community_infomap()

    part_q={};
#    print Q
    for i,mem in enumerate(Q.membership):
            part_q[i]=[];
            part_q[i]=mem;
    #print 'Number of clusters = ', len(list(set(part_q.values())));
    return part_q;


def fixed_delta_t_temporal_partition(TG, delta_t,mode='modularity',verbose=False):
    T_partition={};
    for t in range(0,len(TG)-delta_t, delta_t):
        T_partition[t]=[];
        if verbose==True:
            print 'Calculating infomap of slice ', t; 
        if mode=='modularity':
            T_partition[t]=modularity_partition(TG,t,t+delta_t);
        elif mode=='infomap':
            T_partition[t]=infomap_partition(TG,t,t+delta_t);
        elif mode=='fastgreedy':
            print 'Greedy modularity chosen.'
            T_partition[t]=fast_modularity_partition(TG,t,t+delta_t);
    return T_partition;


def fixed_tau_m_series_partition(M_series,sm,w,tau,mode='multilevel',verbose=False):
    tau_partition={}
    for t in range(max(M_series.keys())):
        if verbose==True:
            print t;
        tau_partition[t]=[];

        M_disposable=np.eye(len(M_series[t]));
        for i in range(tau+1):
            M_disposable=np.dot(M_disposable,M_series[(t+i)%len(M_series)]);
        M_disposable=np.dot(np.diag(w,0),M_disposable);# - sm;
        print M_disposable
        g=igraph.Graph.Weighted_Adjacency(list(np.array(M_disposable) + np.array(M_disposable).transpose()), attr="weight", loops=True);
        g.to_undirected(combine_edges='mean');
        print g.is_directed()
        if mode=='multilevel':
            tau_partition[t]=g.community_multilevel(weights='weight');
        if mode=='fast_greedy':
            tau_partition[t]=g.community_fastgreedy(weights='weight');
        if mode=='leading_eigenvector':
            tau_partition[t]=g.community_leading_eigenvector();
        if mode=='optimal_modularity':
            tau_partition[t]=g.community_optimal_modularity(verbose=True);
        if mode=='spinglass':
            tau_partition[t]=g.community_spinglass(weights='weight');
        if mode=='infomap':
            tau_partition[t]=g.community_infomap(edge_weights='weight');

    return tau_partition;



def fixed_tau_m_series_partition_no_igraph(M_series,sm,w,tau,verbose=False):
    import community;
    import matplotlib.pyplot as plt
    tau_partition={}
    for t in range(max(M_series.keys())):
        if verbose==True:
            print t;
        tau_partition[t]=[];
        M_disposable=np.eye(len(M_series[t]));
        for i in range(tau+1):
            M_disposable=np.dot(M_disposable,M_series[(t+i)%len(M_series)]);
        M_disposable=np.dot(np.diag(w,0),M_disposable);# - sm;
        g=nx.to_networkx_graph(M_disposable+M_disposable.T - np.diag(M_disposable.diagonal()) ,create_using=nx.Graph()); 
        #print g.number_of_nodes(), g.number_of_edges(), sum(g.degree().values())
        tau_partition[t] = community.best_partition(g);
        if verbose==True:
            plt.figure, plt.pcolor(np.array(nx.to_numpy_matrix(g))), plt.colorbar();
            plt.show()
            print tau_partition[t]
    return tau_partition;

def t_delta_matrix(mat,M_series,t,tau,verbose=False):    
    if verbose==True:
        print t;
    t_delta_matrix=mat;
    t_delta_matrix=np.dot(t_delta_matrix,M_series[(t+tau)%len(M_series)]);
    #t_delta_matrix=np.dot(np.diag(w,0),t_delta_matrix);# - sm;
    return t_delta_matrix;

def t_delta_partition(t_delta_matrix,sm,verbose=False):
    import community;
    g=nx.to_networkx_graph(t_delta_matrix+t_delta_matrix.T - np.diag(t_delta_matrix.diagonal()) ,create_using=nx.Graph()); 
    if verbose==True:
        plt.figure, plt.pcolor(np.array(nx.to_numpy_matrix(g))), plt.colorbar();
        plt.show()
    return community.best_partition(g);


def VI_quiver_plot(partitions_dict,verbose=False):
    from pylab import *
    from numpy import ma
    import igraph as ig
    U=np.zeros((len(partitions_dict),len(partitions_dict[0])));
    V=np.zeros((len(partitions_dict),len(partitions_dict[0])));
    X=np.zeros((len(partitions_dict),len(partitions_dict[0])));
    Y=np.zeros((len(partitions_dict),len(partitions_dict[0])));
    
    n_deltas=len(partitions_dict);
    for i,delta in enumerate(sorted(partitions_dict.keys())):
        for j,t in enumerate(sorted(partitions_dict[delta].keys())):
            X[i,j]=delta;
            Y[i,j]=t;
            try:
                a=(ig.compare_communities(partitions_dict[delta][t].values(), partitions_dict[delta][partitions_dict[delta].keys()[j+1]%max(partitions_dict[delta].keys())].values()));
                b=(ig.compare_communities(partitions_dict[partitions_dict.keys()[(i+1)%max(partitions_dict.keys())]][t].values(), partitions_dict[delta][t].values()));
                if a!=0:
                    U[i,j]=a;
                if b!=0:
                    V[i,j]=b;
            except:
                if verbose==True:
                    print 'Error at:', (i,j),(delta,t)
    
    figure(figsize=(10,10))
    Q = quiver( U, V)
    qk = quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
                   fontproperties={'weight': 'bold'})
    l,r,b,t = axis();
    dx, dy = r-l, t-b
    axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])

    title('Minimal arguments, no kwargs');
    show();

    return X,Y,U,V;




def temporal_stats(real_TG):
    av_real=[];
    for t in real_TG:
        av_real.append(2*float(real_TG[t].number_of_edges())/float(real_TG[t].number_of_nodes()))

    active_nodes=[];
    for t in real_TG:
        count=0;
        deg=nx.degree(real_TG[t])
        for n in deg:
            if deg[n]>0: count+=1 
        active_nodes.append(count)
    print 'Average temporal degree ',mean(av_real)
    print 'Average number of active nodes ', mean(active_nodes)/float(real_TG[t].number_of_nodes())

#########################################################################################################
##################### New code for activity values etc (it works @15/6/2013)
####################################################################################################################

def activity_potential(TG):
    activity_V={}
    for t in TG:
        count=0;
        deg=nx.degree(TG[t])
        for n in deg:
            if deg[n]>0: 
                if n in activity_V:
                    activity_V[n]+=1;
                else:
                     activity_V[n]=1;
    for n in activity_V:
        activity_V[n]=float(activity_V[n])/float(len(TG));
    return activity_V;

def eta_fitter(activity_V,average_degree,m):
    av_x=np.mean(activity_V.values());
    return float(average_degree)/float(av_x * m);
    
def final_activity_dict(activity_V,eta):
    activity_dict={}
    for n in activity_V:
        activity_dict[n]=activity_V[n]*eta;
    return activity_dict;


def activity_model_random_graph(activity_dict,m=2,T=100):
    import networkx as nx 
    import sys, os
    import pickle
    import random
    
    # temporal graph dictionary
    TG={};
    nodes=activity_dict.keys();
    for t in range(T):
        g=nx.Graph();
        g.add_nodes_from(nodes);
        for node in activity_dict:
            if random.random()<=activity_dict[node]:
                # activation!
                nodes_removed=activity_dict.keys();
                nodes_removed.remove(node);
                new_neighbours=[];
                for i in range(m):
                    new_neighbours.append(random.choice(nodes_removed));
                    g.add_edge(node,new_neighbours[-1]);
                    nodes_removed.remove(new_neighbours[-1]);
        TG[t]=[];   
        TG[t]=g;
    
    return TG;

def edge_activation_output_matrix(TG):
    edgelist=[];
    for t in TG:
        edgelist.extend(TG[t].edges());
    edgelist=list(set(edgelist));

    edge_activation_matrix=nos((len(edgelist),len(TG)),dtype=np.int);
    for i,edge in enumerate(edgelist):
        for t in TG:
            if TG[t].has_edge(edge[0],edge[1]):
                edge_activation_matrix[i,t]=1;
    return edge_activation_matrix,edgelist;




