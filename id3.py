import ast
import csv
import sys
import math
import os
import pydot

global tree,ct
tree =pydot.Dot(graph_type='digraph')
ct=0

class node:
	name=""
	isLeaf=False
	values={}

def load_csv_to_header_data(filename):
    path = os.path.normpath(os.getcwd() + filename)
    fs = csv.reader(open(path))
    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    # print(headers)
    for i in range(0, len(headers)):
    	headers[i]=headers[i].strip()
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)

    data = {
        'header': headers,
        'rows': all_row[1:],
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }
    return data

def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx

def get_uniq_values(data):
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()

    val_map = {}
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]
            if val not in val_map.keys():
                val_map[att_name].add(val)
    return val_map

def get_class_labels(data, target_attribute):
    rows = data['rows']
    col_idx = data['name_to_idx'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_idx]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1
    return labels

def entropy(n, labels):
    ent = 0
    for label in labels.keys():
        p_x = labels[label] / n
        ent += - p_x * math.log(p_x, 2)
    return ent

def partition_data(data, group_att):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }
            # print(row_val)
        partitions[row_val]['rows'].append(row)
    return partitions

def avg_entropy_w_partitions(data, splitting_att, target_attribute):
    # find uniq values of splitting att
	data_rows = data['rows']
	n = len(data_rows)
	partitions = partition_data(data, splitting_att)

	avg_ent = 0

	for partition_key in partitions.keys():
		partitioned_data = partitions[partition_key]
		partition_n = len(partitioned_data['rows'])
		partition_labels = get_class_labels(partitioned_data, target_attribute)
		partition_entropy = entropy(partition_n, partition_labels)
		avg_ent += partition_n / n * partition_entropy
	# print(partition_key)
	return avg_ent, partitions

def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl

def id3(data, uniqs, remaining_atts, target_attribute):
	global tree,ct

	labels = get_class_labels(data, target_attribute)
	root =node()
	par_node=pydot.Node("ID3",style="filled",fillcolor="#42597a")
	if len(labels.keys()) == 1:
		root.isLeaf=True
		root.name=labels.keys[0].strip()
		par_node=pydot.Node('node%d'%ct,style="filled",shape='box',fillcolor="#42597a",label=root.name)
		tree.add_node(par_node)
		ct+=1
		return root																		

	if len(remaining_atts) == 0:
		root.name=most_common_label(labels).strip()
		par_node=pydot.Node('node%d' %ct,style="filled",shape='box',fillcolor="#42597a",label=root.name)
		tree.add_node(par_node)
		ct+=1
		root.isLeaf=True
		return root

	n = len(data['rows'])
	ent = entropy(n, labels)

	max_info_gain = None
	max_info_gain_att = None
	max_info_gain_partitions = None

	for remaining_att in remaining_atts:
		avg_ent, partitions = avg_entropy_w_partitions(data, remaining_att, target_attribute)
		# print(partitions.keys())
		info_gain = ent - avg_ent
		if max_info_gain is None or info_gain > max_info_gain:
		    max_info_gain = info_gain
		    max_info_gain_att = remaining_att
		    max_info_gain_partitions = partitions
	# print(max_info_gain_att)
	remaining_atts_for_subtrees = set(remaining_atts)
	remaining_atts_for_subtrees.discard(max_info_gain_att)
	uniq_att_values = uniqs[max_info_gain_att]

	root.name=max_info_gain_att
	root.isLeaf=False
	par_node=pydot.Node(root.name,style="filled",fillcolor="#858c96")
	# ct+=1
	tree.add_node(par_node)

	for partition_key in max_info_gain_partitions.keys():
		# print(partition_key)
		rows=max_info_gain_partitions[partition_key]['rows']
		tar_att_idx = data['name_to_idx'][target_attribute]
		tar_attr=rows[0][tar_att_idx]
		flg=0
		for row in rows:
			if(row[tar_att_idx]!=tar_attr):
				flg=1
				break
		if(flg==0):
			child=node()
			child.isLeaf=True
			child.name=tar_attr
			root.values[partition_key.strip()]=child
			child_node=pydot.Node('node%d' %ct,style="filled",shape='box',fillcolor="#42597a",label=child.name)
			ct+=1
			tree.add_node(child_node)
			tree.add_edge(pydot.Edge(par_node,child_node,label=partition_key,color="blue"))
		else:
			remaining_data= max_info_gain_partitions[partition_key]
			child=id3(remaining_data, uniqs, remaining_atts_for_subtrees, target_attribute)
			root.values[partition_key.strip()]=child
			child_node=pydot.Node(child.name,style="filled",fillcolor="#858c96")
			# ct+=1
			tree.add_node(child_node)
			tree.add_edge(pydot.Edge(par_node,child_node,label=partition_key,color="blue"))

	return root


def testUtil(root,inp,g):

	if(root.isLeaf):
		return root.name

	key=root.name
	par_node=pydot.Node(key,style="filled",fillcolor="#336ec4")
	g.add_node(par_node)
	value=inp[key]
	# print(key," ",value)
	if(root.values.get(value)!=None):
		root=root.values[value]

	res=testUtil(root,inp,g)
	if(root.isLeaf==False):
		child_node=pydot.Node(root.name,style="filled",fillcolor="#336ec4")
	else:
		child_node=pydot.Node(root.name,style="filled",shape='box',fillcolor="green")
	g.add_node(child_node)
	g.add_edge(pydot.Edge(par_node,child_node,label=value,color="blue"))
	return res

def test(root):
	fp=open("queries.txt","r")
	st=fp.readlines()
	length= len(st)
	inp={}
	header=["Stream","Slope","Elevation","Vegetation"]
	
	for i in range(length):
		st[i]=st[i].strip('\n')
		dt=st[i].split()
		for j in range(len(dt)):
			inp[header[j]]=dt[j]
		# print(root.name)
		name="query_"+str(i+1)
		g=pydot.Dot(graph_type='digraph')
		res=testUtil(root,inp,g)
		print(res)
		g.write_png('%s.png'%(name))



def main():
    
	data = load_csv_to_header_data("//dataset.csv")
	target_attribute =  'Vegetation'
	remaining_attributes = set(data['header'])
	#print(remaining_attributes)
	remaining_attributes.remove(target_attribute)

	uniqs = get_uniq_values(data)

	root = id3(data, uniqs, remaining_attributes, target_attribute)
	global tree
	tree.write_png('id3.png')
	#print("mnit")
	# pretty_print_tree(root)
	# print(root.values['medium'].values)


	test(root)

main()

