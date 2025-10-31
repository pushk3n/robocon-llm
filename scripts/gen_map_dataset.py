import xml.etree.ElementTree as ET
import json
import random
import networkx as nx
import os
from collections import defaultdict

def create_graph_from_xml(xml_root):
    """
    从XML根节点解析地图数据.
    - G: 一个 networkx.Graph 对象, 存储拓扑连接.
    - nodes_info: 字典, 存储每个节点的详细标签 (k-v pairs).
    - name_to_id: 字典, 从区域名称映射到ID.
    - hierarchy: 字典, 存储层次关系 (e.g., {'MC': ['1', '2', ...]})
    """
    G = nx.Graph()
    nodes_info = {}
    name_to_id = {}
    hierarchy = defaultdict(list)
    rules = {}

    # 1. 解析所有节点 (Areas)
    for node_elem in xml_root.findall('node'):
        node_id = node_elem.get('id')
        tags = {tag.get('k'): tag.get('v') for tag in node_elem.findall('tag')}
        nodes_info[node_id] = tags
        G.add_node(node_id)
        
        if 'name' in tags:
            name_to_id[tags['name']] = node_id

    # 2. 解析所有边 (Passages)
    for way_elem in xml_root.findall('way'):
        refs = [nd.get('ref') for nd in way_elem.findall('nd')]
        if len(refs) == 2:
            # 确保两个节点都存在于图中
            if refs[0] in G and refs[1] in G:
                G.add_edge(refs[0], refs[1])
            else:
                print(f"警告: Way {way_elem.get('id')} 引用了不存在的节点: {refs}")

    # 3. 解析层次结构和规则 (Relations)
    for rel_elem in xml_root.findall('relation'):
        rel_tags = {tag.get('k'): tag.get('v') for tag in rel_elem.findall('tag')}
        
        # 解析层次结构
        if rel_tags.get('type') == 'hierarchy':
            parent_area_elem = rel_elem.find("member[@role='parent_area']")
            if parent_area_elem is not None:
                parent_id = parent_area_elem.get('ref')
                parent_name = nodes_info.get(parent_id, {}).get('name', parent_id)
                for member in rel_elem.findall("member[@role='sub_area']"):
                    hierarchy[parent_name].append(member.get('ref'))
        
        # 解析规则
        if rel_tags.get('id') == '3001' or rel_tags.get('type') == 'metadata':
            for tag in rel_elem.findall('tag'):
                if tag.get('k').startswith('rule:'):
                    rules[tag.get('k')] = tag.get('v')

    return G, nodes_info, name_to_id, hierarchy, rules

def generate_qa_pairs(xml_file_path, output_path, num_samples_per_type=100):
    """
    读取XML文件并生成多样化的问答对.
    """
    
    # --- 1. 加载和解析数据 ---
    
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建绝对路径
    abs_xml_path = os.path.join(script_dir, xml_file_path)
    abs_output_path = os.path.join(script_dir, output_path)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    try:
        # 读取完整的XML文件内容作为 "input" (上下文)
        with open(abs_xml_path, 'r', encoding='utf-8') as f:
            map_context = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到XML文件: {abs_xml_path}")
        print("请确保脚本位于 'scripts' 目录下, 并且 'data/initial/map.xml' 存在.")
        return

    # 解析XML
    tree = ET.parse(abs_xml_path)
    root = tree.getroot()
    
    G, nodes_info, name_to_id, hierarchy, rules = create_graph_from_xml(root)
    
    all_node_ids = list(nodes_info.keys())
    qa_pairs = []

    # --- 2. 生成QA对 ---

    # 类型1: 属性查询 (Attribute Questions)
    for _ in range(num_samples_per_type):
        node_id = random.choice(all_node_ids)
        node_name = nodes_info[node_id].get('name', f"ID {node_id}")
        
        # 随机选择一个属性进行提问
        if not nodes_info[node_id]:
            continue
        
        attr_key = random.choice(list(nodes_info[node_id].keys()))
        attr_value = nodes_info[node_id][attr_key]
        
        question = f"区域 '{node_name}' (ID: {node_id}) 的 '{attr_key}' 属性是什么？"
        answer = f"区域 '{node_name}' (ID: {node_id}) 的 '{attr_key}' 属性是 '{attr_value}'。"
        
        qa_pairs.append({
            "instruction": question, 
            "input": map_context, 
            "output": answer
        })

    # 类型2: R2可达性查询 (R2 Accessibility Questions)
    for _ in range(num_samples_per_type):
        start_node_id = random.choice(all_node_ids)
        end_node_id = random.choice(all_node_ids)
        
        start_name = nodes_info[start_node_id].get('name', f"ID {start_node_id}")
        end_name = nodes_info[end_node_id].get('name', f"ID {end_node_id}")

        question = f"我是一个R2机器人，我是否可以从 '{start_name}' (ID: {start_node_id}) 移动到 '{end_name}' (ID: {end_node_id})？"
        
        # 检查R2访问权限 (默认为 'true')
        can_access = nodes_info[end_node_id].get('R2_access', 'true') == 'true'
        
        if not can_access:
            answer = f"不可以。目标区域 '{end_name}' (ID: {end_node_id}) 的 'R2_access' 属性为 'false'，R2机器人禁止进入。"
        else:
            try:
                path_exists = nx.has_path(G, start_node_id, end_node_id)
                if path_exists:
                    answer = f"可以。从 '{start_name}' 到 '{end_name}' 存在拓扑路径，并且目标区域 '{end_name}' 允许R2机器人进入。"
                else:
                    answer = f"不可以。虽然目标区域 '{end_name}' 允许R2机器人进入，但在地图拓扑上不存在从 '{start_name}' 到达的路径。"
            except nx.NodeNotFound:
                 answer = f"错误：地图中找不到起点 '{start_name}' (ID: {start_node_id}) 或终点 '{end_name}' (ID: {end_node_id})。"

        qa_pairs.append({
            "instruction": question, 
            "input": map_context, 
            "output": answer
        })

    # 类型3: 最短路径查询 (Shortest Path Questions)
    for _ in range(num_samples_per_type):
        start_node_id, end_node_id = random.sample(all_node_ids, 2)
        start_name = nodes_info[start_node_id].get('name', f"ID {start_node_id}")
        end_name = nodes_info[end_node_id].get('name', f"ID {end_node_id}")

        question = f"从 '{start_name}' (ID: {start_node_id}) 到 '{end_name}' (ID: {end_node_id}) 的最短拓扑路径是什么？"
        
        try:
            path = nx.shortest_path(G, source=start_node_id, target=end_node_id)
            path_names = [nodes_info[nid].get('name', f"ID {nid}") for nid in path]
            answer = f"从 '{start_name}' 到 '{end_name}' 的最短拓扑路径是：{' -> '.join(path_names)}。"
        except nx.NetworkXNoPath:
            answer = f"在地图上不存在从 '{start_name}' 到 '{end_name}' 的连接路径。"
        
        qa_pairs.append({
            "instruction": question, 
            "input": map_context, 
            "output": answer
        })

    # 类型4: 规则查询 (Rule Questions)
    if rules:
        rule_keys = list(rules.keys())
        for _ in range(num_samples_per_type // 2): # 规则问题少一些
            rule_key = random.choice(rule_keys)
            rule_text = rules[rule_key]
            
            question = f"地图规则 '{rule_key}' 的内容是什么？"
            answer = f"规则 '{rule_key}' 的内容是：{rule_text}"
            
            qa_pairs.append({
                "instruction": question, 
                "input": map_context, 
                "output": answer
            })

    # 类型5: 层次结构查询 (Hierarchy Questions)
    if hierarchy:
        parent_names = list(hierarchy.keys())
        for _ in range(num_samples_per_type // 2): # 层次问题少一些
            parent_name = random.choice(parent_names)
            sub_area_ids = hierarchy[parent_name]
            sub_area_names = [nodes_info[nid].get('name', f"ID {nid}") for nid in sub_area_ids]
            
            question = f"'{parent_name}' 区域包含了哪些子区域？"
            answer = f"'{parent_name}' 区域包含以下子区域：{', '.join(sub_area_names)}。"
            
            qa_pairs.append({
                "instruction": question, 
                "input": map_context, 
                "output": answer
            })

    # --- 3. 保存文件 ---
    
    # 随机打乱数据集
    random.shuffle(qa_pairs)
    
    # 保存为 .jsonl 文件
    with open(abs_output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            # 使用 ensure_ascii=False 来正确处理中文字符
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
    print(f"成功！已生成 {len(qa_pairs)} 条问答数据到 {abs_output_path}")

if __name__ == "__main__":
    # 路径相对于此脚本 (scripts/prepare_dataset.py)
    xml_file_path = '../data/initial/map.xml'
    jsonl_output_path = '../data/processed/training_dataset.jsonl'
    
    # 每种类型的问题生成约100条, 总共约 400-500 条
    generate_qa_pairs(xml_file_path, jsonl_output_path, num_samples_per_type=100)

