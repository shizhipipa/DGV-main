import subprocess

def create_digraph(name, nodes):
    with open('digraph.gv', 'w') as f:
        f.write('digraph {\n')
        digraph = ''
        for n_id, node in nodes.items():
            label = node.get_code() + '\n' + node.label
            f.write(f'"{node.get_code()}" [label="{label}"]\n')
            for e_id, edge in node.edges.items():
                if edge.type != 'Ast':
                    continue
                if edge.node_in in nodes and edge.node_in != node.id:
                    n_in = nodes.get(edge.node_in)
                    label = '"' + n_in.label + '"'
                    digraph += f'"{node.get_code()}" -> "{n_in.get_code()}" [label={label}]\n'
                '\n\t\t\t\tif edge.node_out in nodes and edge.node_out != node.id:\n\t\t\t\t\tn_out = nodes.get(edge.node_out)\n\t\t\t\t\tlabel = """ + n_out.label + """\n\t\t\t\t\tdigraph += f""{n_out.get_code()}" -> "{node.get_code()}" [label={label}]\n"\t\t\n\t\t\t\t'
        f.write(digraph)
        f.write('}\n')
    subprocess.run(['dot', '-Tps', 'digraph.gv', '-o', f'{name}.ps'], shell=False)
'\ndef to_digraph(name, nodes):\n    k_nodes = nodes.keys()\n    code = {}\n    connections = { "in" : dict.fromkeys(k_nodes), "out" : dict.fromkeys(k_nodes) }\n\n\tfor n_id, node in nodes.items():\n\t\t#print(n_id, node.properties)\n\t\tconnections = node.connections(connections, "Ast")\n\t\tcode.update({n_id : node.get_code()})\n\n\tcreate_digraph(name, code, k_nodes, connections)\n'
