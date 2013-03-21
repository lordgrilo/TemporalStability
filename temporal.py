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

def timevarying_graph_generator(data, edgelist,dim):
    print data.shape, edgelist.shape
    tempG={};
    if dim!=0:
        data=np.transpose(data);
    print data.shape
    T=data.shape[1];
    print T
    primal_graph=nx.read_edgelist(edgelist_file);
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

def instant_partition_matrix(partition):
    partition_index={};
    for i,p in enumerate(list(set(partition.values()))):
        partition_index[p]=i;
    N=len(list(set(partition.keys())));
    num_comms=len(list(set(partition.values())));        
    H=np.zeros((N,num_comms));
    print H.shape;
    for node in partition:
        H[node][partition_index[partition[node]]]=1;
    return H; 


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
        #start of loop on physical times
        H=instant_partition_matrix(temporal_partition[partition_temporal_swaps[current_partition_time]]);
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
        proj_R_tau[tau] = [];
        proj_R_tau[tau] = np.mean(tr_av);
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
    nx.write_pajek(new_graph,'temp_graph.net');
    g=ig.read('temp_graph.net',format="pajek")
    os.remove('temp_graph.net');
    Q=g.community_spinglass()
    part_q={};
 #   print Q
    for i,mem in enumerate(Q.membership):
            part_q[i]=[];
            part_q[i]=mem;
    return part_q;

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
    print new_graph.number_of_nodes();
    nx.write_pajek(new_graph,'temp_graph.net');
    g=ig.read('temp_graph.net',format="pajek")
    os.remove('temp_graph.net');
    Q=g.community_optimal_modularity();
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
    print new_graph.number_of_nodes();
    nx.write_pajek(new_graph,'temp_graph.net');
    g=ig.read('temp_graph.net',format="pajek")
    os.remove('temp_graph.net');
    Q=g.community_infomap();
    part_q={};
#    print Q
    for i,mem in enumerate(Q.membership):
            part_q[i]=[];
            part_q[i]=mem;
    print 'Number of clusters = ', len(list(set(part_q.values())));
    return part_q;


def fixed_delta_t_temporal_partition(TG, delta_t):
    T_partition={};
    for t in range(0,len(TG)-delta_t, delta_t):
        T_partition[t]=[];
        print 'Calculating infomap of slice ', t; 
        T_partition[t]=infomap_partition(TG,t,t+delta_t);
    return T_partition;

